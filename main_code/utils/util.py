import cv2
import torch
import h5py
import math
import shutil
import os
import copy
import numpy as np
import torch.nn as nn
from scipy.optimize import fminbound
from skimage.measure import compare_ssim as ssim

evaluateSnr = lambda x, xhat: 20*np.log10(np.linalg.norm(x.flatten('F'))/np.linalg.norm(x.flatten('F')-xhat.flatten('F')))

def optimizeTau(x, algoHandle, taurange, maxfun=20):
    # maxfun ~ number of iterations for optimization
    fun = lambda tau: -psnr(algoHandle(tau)[0], x)
    tau = fminbound(fun, taurange[0],taurange[1], xtol = 1e-3, maxfun = maxfun, disp = 3)
    return tau


def h5py2mat(data):
    result = np.array(data)
    print(result.shape)

    if len(result.shape) == 3 and result.shape[0] > result.shape[1]:
        result = result.transpose([1,0,2])
    elif len(result.shape) == 3 and result.shape[1] < result.shape[2]:
        result = result.transpose([2,1,0])
    elif len(result.shape) == 3 and result.shape[1] > result.shape[2]:
        result = result.transpose([2,1,0])        
    print(result.shape)    
    return result

def complex_multiple_torch(x: torch.Tensor, y: torch.Tensor):
    x_real, x_imag = torch.unbind(x, -1)
    y_real, y_imag = torch.unbind(y, -1)

    res_real = torch.mul(x_real, y_real) - torch.mul(x_imag, y_imag)
    res_imag = torch.mul(x_real, y_imag) + torch.mul(x_imag, y_real)

    return torch.stack([res_real, res_imag], -1)

###################
# Read Images
###################
def np2torch_complex(array: np.ndarray):
    return torch.stack([torch.from_numpy(array.real), torch.from_numpy(array.imag)], -1)


def addwgn_torch(x: torch.Tensor, inputSnr):
    noiseNorm = torch.norm(x.flatten() * 10 ** (-inputSnr / 20))

    # xBool = np.isreal(x)
    # real = True
    # for e in np.nditer(xBool):
    #     if not e:
    #         real = False
    # if real:
    #     noise = np.random.randn(np.shape(x)[0], np.shape(x)[1])
    # else:
    #     noise = np.random.randn(np.shape(x)[0], np.shape(x)[1]) + \
    #         1j * np.random.randn(np.shape(x)[0], np.shape(x)[1])

    noise = torch.randn(x.shape[-2], x.shape[-1])
    noise = noise / torch.norm(noise.flatten()) * noiseNorm
    
    rec_y = x + noise.cuda()

    return rec_y

def compare_snr(img_test, img_true):
    return 20 * torch.log10(torch.norm(img_true.flatten()) / torch.norm(img_true.flatten() - img_test.flatten()))



def rsnr_cal(rec,oracle):
    "regressed SNR"
    sumP    =        sum(oracle.reshape(-1))
    sumI    =        sum(rec.reshape(-1))
    sumIP   =        sum( oracle.reshape(-1) * rec.reshape(-1) )
    sumI2   =        sum(rec.reshape(-1)**2)
    A       =        np.matrix([[sumI2, sumI],[sumI, oracle.size]])
    b       =        np.matrix([[sumIP],[sumP]])
    c       =        np.linalg.inv(A)*b #(A)\b
    rec     =        c[0,0]*rec+c[1,0]
    err     =        sum((oracle.reshape(-1)-rec.reshape(-1))**2)
    SNR     =        10.0*np.log10(sum(oracle.reshape(-1)**2)/err)

    if np.isnan(SNR):
        SNR=0.0
    return SNR

def compute_rsnr(x, xhat):
    if len(x.shape) == 2:
        avg_rsnr = rsnr_cal(xhat, x)
    elif len(x.shape) == 3 and x.shape[0] < x.shape[1]:   
        rsnr = np.zeros([1,x.shape[0]])
        for num_imgs in range(0,x.shape[0]):
            rsnr[:,num_imgs] = rsnr_cal(xhat, x)
        avg_rsnr = np.mean(rsnr)
    return avg_rsnr

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant_(m.bias.data, 0.0)

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight.data)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant_(m.bias.data, 0.0)

def copytree(src=None, dst=None, symlinks=False, ignore=None):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            copytree(s, d, symlinks, ignore)
        else:
            if not os.path.exists(d) or os.stat(s).st_mtime - os.stat(d).st_mtime > 1:
                shutil.copy2(s, d)
                
def data_augmentation(image, mode):
    out = image
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return out                



def rgb2ycbcr(rgb):
    m = np.array([[ 65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112],
                  [ 112, -93.786, -18.214]])
    shape = rgb.shape
    if len(shape) == 3:
        rgb = rgb.reshape((shape[0] * shape[1], 3))
    ycbcr = np.dot(rgb, m.transpose() / 255.)
    ycbcr[:,0] += 16.
    ycbcr[:,1:] += 128.
    return ycbcr.reshape(shape)

# ITU-R BT.601
# https://en.wikipedia.org/wiki/YCbCr
# YUV -> RGB
def ycbcr2rgb(ycbcr):
    m = np.array([[ 65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112],
                  [ 112, -93.786, -18.214]])
    shape = ycbcr.shape
    if len(shape) == 3:
        ycbcr = ycbcr.reshape((shape[0] * shape[1], 3))
    rgb = copy.deepcopy(ycbcr)
    rgb[:,0] -= 16.
    rgb[:,1:] -= 128.
    rgb = np.dot(rgb, np.linalg.inv(m.transpose()) * 255.)
    return rgb.clip(0, 255).reshape(shape)

def imread_CS_py(Iorg):
    block_size = 33
    [row, col] = Iorg.shape
    row_pad = block_size-np.mod(row,block_size)
    col_pad = block_size-np.mod(col,block_size)
    Ipad = np.concatenate((Iorg, np.zeros([row, col_pad])), axis=1)
    Ipad = np.concatenate((Ipad, np.zeros([row_pad, col+col_pad])), axis=0)
    [row_new, col_new] = Ipad.shape

    return [Iorg, row, col, Ipad, row_new, col_new]


def img2col_py(Ipad, block_size):
    [row, col] = Ipad.shape
    row_block = row/block_size
    col_block = col/block_size
    block_num = int(row_block*col_block)
    img_col = np.zeros([block_size**2, block_num])
    count = 0
    for x in range(0, row-block_size+1, block_size):
        for y in range(0, col-block_size+1, block_size):
            img_col[:, count] = Ipad[x:x+block_size, y:y+block_size].reshape([-1])
            # img_col[:, count] = Ipad[x:x+block_size, y:y+block_size].transpose().reshape([-1])
            count = count + 1
    return img_col


def col2im_CS_py(X_col, row, col, row_new, col_new):
    block_size = 33
    X0_rec = np.zeros([row_new, col_new])
    count = 0
    for x in range(0, row_new-block_size+1, block_size):
        for y in range(0, col_new-block_size+1, block_size):
            X0_rec[x:x+block_size, y:y+block_size] = X_col[:, count].reshape([block_size, block_size])
            # X0_rec[x:x+block_size, y:y+block_size] = X_col[:, count].reshape([block_size, block_size]).transpose()
            count = count + 1
    X_rec = X0_rec[:row, :col]
    return X_rec


def imread_CS_torch(Iorg):
    block_size = 33
    [row, col] = Iorg.shape
    row_pad = block_size-np.mod(row,block_size)
    col_pad = block_size-np.mod(col,block_size)
    Ipad = torch.cat((Iorg, torch.zeros([row, col_pad]).cuda()), axis=1)
    Ipad = torch.cat((Ipad, torch.zeros([row_pad, col+col_pad]).cuda()), axis=0)
    [row_new, col_new] = Ipad.shape

    return [Iorg, row, col, Ipad, row_new, col_new]

def img2col_torch(Ipad, block_size):
    [row, col] = Ipad.shape
    row_block = row/block_size
    col_block = col/block_size
    block_num = int(row_block*col_block)
    img_col = torch.zeros([block_size**2, block_num]).cuda()
    count = 0
    for x in range(0, row-block_size+1, block_size):
        for y in range(0, col-block_size+1, block_size):
            img_col[:, count] = Ipad[x:x+block_size, y:y+block_size].reshape([-1])
            # img_col[:, count] = Ipad[x:x+block_size, y:y+block_size].transpose().reshape([-1])
            count = count + 1
    return img_col

def col2im_CS_torch(X_col, row, col, row_new, col_new):
    block_size = 33
    X0_rec = torch.zeros([row_new, col_new]).cuda()
    count = 0
    for x in range(0, row_new-block_size+1, block_size):
        for y in range(0, col_new-block_size+1, block_size):
            X0_rec[x:x+block_size, y:y+block_size] = X_col[:, count].reshape([block_size, block_size])
            # X0_rec[x:x+block_size, y:y+block_size] = X_col[:, count].reshape([block_size, block_size]).transpose()
            count = count + 1
    X_rec = X0_rec[:row, :col]
    return X_rec

def psnr(img1, img2):
    img1.astype(np.float32)
    img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))    