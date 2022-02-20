import os
import json
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as utils
import torch.nn.functional as F
from torch.nn import init
import torch.nn as nn
import torch
from utils.util import *
from skimage.measure import compare_ssim as ssim
from collections import OrderedDict
from argparse import ArgumentParser
from datetime import datetime
import scipy.io as sio
from tqdm import tqdm
import numpy as np
import platform
import glob
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

with open('config.json') as File:
    config = json.load(File)
print('You are currently using GPU: ', config['setting']['gpu_index'])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = config['setting']['gpu_index']


now = datetime.now()
root_path = config['root_path']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print_flag = config['setting']['print']   # print parameter number
start_epoch = config['train']['start_epoch']
end_epoch = config['train']['end_epoch']
learning_rate = config['train']['inti_lr']
nrtrain = config['train']['num_train']   # number of training blocks
batch_size = config['train']['batch_size']
layer_num = config['unfold_model']['num_iter']
n_channels = config['cnn_model']['n_channels']

cs_ratio = config["dataset"]['train']['CS_ratio']
n_output = config["dataset"]['train']['IMG_Patch'] ** 2
# n_input = ratio_dict[cs_ratio]

save_dir = root_path + "Experiements/Unfold_prior_MRI" + \
    "/Conv/%s_CS_SN-DnCNN_artifact_layer_%d_ratio_%d_lr_%.4f_WS" % (
        str(now.strftime("%d-%b-%Y-%H-%M-%S")), layer_num, cs_ratio, learning_rate)

copytree(src=config['code_path'], dst=save_dir)
writer = SummaryWriter(log_dir=os.path.join(save_dir, 'log'))

# Load CS Sampling Matrix: phi
Phi_data_Name = '%s/mask_%d.mat' % (config["dataset"]
                                    ['Phi_datapath'], cs_ratio)
Phi_data = sio.loadmat(Phi_data_Name)
mask_matrix = Phi_data['mask_matrix']

mask_matrix = torch.from_numpy(mask_matrix).type(torch.FloatTensor)
mask = torch.unsqueeze(mask_matrix, 2)
mask = torch.cat([mask, mask], 2)
mask = mask.to(device)

Training_data = sio.loadmat(config["dataset"]['train_datapath'])
Training_labels = Training_data['labels']

filepaths_test = glob.glob(config["dataset"]['valid_datapath'] + '/*.png')

ImgNum = len(filepaths_test)

# def nesterov(f, x0, lam=1., max_iter=150, tol=1e-8):
#     """ Nesterov acceleration for fixed point iteration. """
#     res = []

#     x = x0
#     s = x.clone()
#     t = torch.tensor(lam, dtype=torch.float32)

#     for k in range(max_iter):
#         xnext = f(s)
#         # res.append((xnext - s).norm().item())
#         tnext = 0.5*(1+torch.sqrt(1+4*t*t))
#         s = xnext + ((t-1)/tnext)*(xnext-x)

#         # update
#         t = tnext
#         x = xnext

#     return x, res

# def anderson(f, x0, m=5, lam=1e-4, max_iter=150, tol=1e-2, beta = 1):
#     """ Anderson acceleration for fixed point iteration. """
#     bsz, d, H, W = x0.shape
#     X = torch.zeros(bsz, m, d*H*W, dtype=x0.dtype, device=x0.device)
#     F = torch.zeros(bsz, m, d*H*W, dtype=x0.dtype, device=x0.device)
#     X[:,0], F[:,0] = x0.view(bsz, -1), f(x0).view(bsz, -1)
#     X[:,1], F[:,1] = F[:,0], f(F[:,0].view_as(x0)).view(bsz, -1)

#     H = torch.zeros(bsz, m+1, m+1, dtype=x0.dtype, device=x0.device)
#     H[:,0,1:] = H[:,1:,0] = 1
#     y = torch.zeros(bsz, m+1, 1, dtype=x0.dtype, device=x0.device)
#     y[:,0] = 1

#     res = []
#     for k in range(2, max_iter):
#         n = min(k, m)
#         G = F[:,:n]-X[:,:n]
#         H[:,1:n+1,1:n+1] = torch.bmm(G,G.transpose(1,2)) + lam*torch.eye(n, dtype=x0.dtype,device=x0.device)[None]
#         alpha = torch.solve(y[:,:n+1], H[:,:n+1,:n+1])[0][:, 1:n+1, 0]   # (bsz x n)

#         X[:,k%m] = beta * (alpha[:,None] @ F[:,:n])[:,0] + (1-beta)*(alpha[:,None] @ X[:,:n])[:,0]
#         F[:,k%m] = f(X[:,k%m].view_as(x0)).view(bsz, -1)
#         res.append((F[:,k%m] - X[:,k%m]).norm().item()/(1e-5 + F[:,k%m].norm().item()))
#         if (res[-1] < tol):
#             break

#     return X[:,k%m].view_as(x0), res

# class CSMRIClass(nn.Module):
#     def __init__(self, mask):
#         super(CSMRIClass, self).__init__()
#         self.mask = mask

#     def grad(self, x, PhiTb, step=None):
#         v = self.FFT_Mask_ForBack(x, self.mask.to(x.device)) - PhiTb
#         return v

#     def fwd_bwd(self, x):
#         v = self.FFT_Mask_ForBack(x, self.mask.to(x.device))
#         return v

#     @staticmethod
#     def FFT_Mask_ForBack(x, mask):
#         with torch.no_grad():
#             x_dim_0 = x.shape[0]
#             x_dim_1 = x.shape[1]
#             x_dim_2 = x.shape[2]
#             x_dim_3 = x.shape[3]
#             x = x.view(-1, x_dim_2, x_dim_3, 1)
#             y = torch.zeros_like(x)
#             z = torch.cat([x, y], 3)
#             fftz = torch.fft(z, 2)
#             z_hat = torch.ifft(fftz * mask.to(x.device), 2)
#             x = z_hat[:, :, :, 0:1]
#             x = x.view(x_dim_0, x_dim_1, x_dim_2, x_dim_3)
#         return x





# # Define ISTA-Net-plus
# class ISTANetplus(nn.Module):
#     def __init__(self, dnn: nn.Module, LayerNo=9):

#         super(ISTANetplus, self).__init__()

#         self.lambda_step = torch.Tensor([1.05]).to(device)
#         self.alpha = torch.Tensor([0.2]).to(device)

#         self.LayerNo = LayerNo
#         self.dnn = dnn

#     def forward(self, PhiTb, mask, accelerate=True):

#         x = PhiTb

#         t = torch.tensor(1., dtype=torch.float32) 
#         s = x.detach().clone()   # gradient update
        
#         fp_dist = []   # for computing symmetric loss

#         for i in range(self.LayerNo):

#             vnext = s - self.lambda_step * (CSMRIClass.FFT_Mask_ForBack(s, mask) - PhiTb)
#             vnext[vnext<=0] = 0
#             znext = self.dnn(vnext)
#             dist = torch.norm(x.flatten() - znext.flatten())**2
#             xnext = (1 - self.alpha) * vnext + self.alpha * znext
#             fp_dist.append(dist)

#             # acceleration
#             if accelerate:
#                 tnext = 0.5*(1+torch.sqrt(1+4*t*t))
#             else:
#                 tnext = 1
#             s = xnext + ((t-1)/tnext)*(xnext-x)
            
#             # update
#             t = tnext
#             x = xnext

#         x_final = x

#         return x_final, fp_dist

# Define RED block
# class Block_RED(nn.Module):

#     def __init__(self, dObj: nn.Module, rObj: nn.Module, tau=1, gamma=1):
#         #gamma_inti=3e-3, tau_inti=1e-1, batch_size=60, device='cuda:0'):

#         super(Block_RED, self).__init__()
#         self.rObj = rObj
#         self.dObj = dObj
#         self.gamma = torch.tensor(tau, dtype=torch.float32)
#         self.tau = torch.tensor(gamma, dtype=torch.float32) 

#     def denoise(self, n_ipt):
#         denoiser = self.rObj(n_ipt)        
#         return denoiser

#     def forward(self, n_ipt:torch.tensor, n_y:torch.tensor, gt=None, **kwargs):

#         delta_g = self.dObj.grad(n_ipt, n_y)
#         xSubD    = torch.abs(self.tau) * (self.rObj(n_ipt))
#         xnext  =  n_ipt - self.gamma * (delta_g.detach() + xSubD) # torch.Size([1, 1, H, W, 2])
#         xnext[xnext<=0] = 0
#         # snr_ = compare_snr(xnext, gt).item()
#         # print(snr_)
#         return xnext

# # Define FixedPoint
# class FixedPoint(nn.Module):
#     def __init__(self, f, solver, **kwargs):
#         super(FixedPoint, self).__init__()
#         self.f = f
#         self.solver = solver
#         self.kwargs = kwargs

#     def forward(self, n_ipt, n_y, gt=None, **grad_kwargs):
#         z, forward_res = self.solver(lambda z : self.f(z, n_y, gt, **grad_kwargs), n_ipt, **self.kwargs)
#         return z, forward_res

# Define DEQFixedPoint
# ##TODO: Try DEQ (unfinished)
# class DEQFixedPoint(nn.Module):
#     def __init__(self, f, solver_img, solver_grad, batch_size, emParams, **kwargs):
#         super().__init__()
#         self.f = f
#         self.solver_img = solver_img
#         self.solver_grad = solver_grad
#         self.kwargs = kwargs

#     def forward(self, n_ipt, n_y=None, gt=None, create_graph=True, only_inputs=True):

#         with torch.no_grad():
#             z, forward_res = self.solver(lambda z : self.f(z, n_y, gt, create_graph=False, only_inputs=False), n_ipt, **self.kwargs)
#         z = self.f(z, n_y, gt, create_graph, only_inputs)
#         # set up Jacobian vector product (without additional forward calls)
#         if create_graph:
#             z0 = z.clone().detach().requires_grad_()
#             r0 = self.f.gamma * self.f.tau * self.f.denoise(z0)
#             def backward_hook(grad):
#                 fTg = lambda y : y - self.f.gamma*self.f.dObj.fwd_bwd(y) - torch.autograd.grad(r0, z0, y, retain_graph=True)[0] + grad
#                 g, self.backward_res = self.solver_grad(fTg, grad, max_iter=65, **self.kwargs)
#                 return g
#             z.register_hook(backward_hook)
#         return z, forward_res

# dObj = CSMRIClass(mask)
# rObj = DnCNN(n_channels=n_channels)
# checkpoint = torch.load(config['keep_training']['load_path'])
# rObj.load_state_dict(checkpoint,strict=True)
# RED = Block_RED(dObj, rObj)
# model = FixedPoint(RED, nesterov, lam=1., max_iter=layer_num).to(device)

network = DnCNN()
network = jacobinNet(network)

if config['setting']['multiGPU']:
    config['setting']['num_gpus'] = torch.cuda.device_count()
    gpu_ids = [t for t in range(config['setting']['num_gpus'])]
    model = nn.DataParallel(network, device_ids=gpu_ids).to(device)

if print_flag:
    num_count = 0
    for para in model.parameters():
        num_count += 1
        print('Layer %d' % num_count)
        print(para.size())

class TrainDataset(Dataset):
    def __init__(self, data, length):
        self.data = data
        self.len = length

    def __getitem__(self, index):
        return torch.Tensor(self.data[index, :]).float()

    def __len__(self):
        return self.len

class ValidDataset(Dataset):
    def __init__(self, datapath):
        self.filepaths_valid = datapath
        
    def __getitem__(self, index):

        imgName = self.filepaths_valid[index]

        Img = cv2.imread(imgName, 0)
        Icol = Img.reshape(1, 256, 256) / 255.0
        Iorg_y = Icol.astype(np.float64)

        Img_output = Iorg_y

        batch_x = torch.from_numpy(Img_output)
        batch_x = batch_x.type(torch.FloatTensor)

        return batch_x

    def __len__(self):
        return len(self.filepaths_valid)

train_loader = DataLoader(dataset=TrainDataset(Training_labels, nrtrain), batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=ValidDataset(filepaths_test), batch_size=8, num_workers=8, shuffle=False, drop_last=True)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-8)

# Training loop

global_step = 0

for epoch_i in range(start_epoch+1, end_epoch+1):

    batch = 0
    avg_snr_training = 0 
    avg_loss_training = 0

    if epoch_i < config['train']['unfold_lr_milstone']:
        current_lr = learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
    elif epoch_i >= config['train']['unfold_lr_milstone'] and epoch_i % 10 == 0:
        current_lr = current_lr *0.7
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

    for data in tqdm(train_loader):

        batch_x = data
        batch_x = batch_x.to(device) # torch.Size([`1`, 256, 256])
        batch_x = batch_x.view(batch_x.shape[0], 1, batch_x.shape[1], batch_x.shape[2]) # torch.Size([2, 1, 256, 256]) #gdt
        noise = torch.FloatTensor(batch_x.size()).normal_(mean=0, std=2/255.) # need to train (2,3,5)/255.
        batch_ipt = batch_x + noise
        # PhiTb = CSMRIClass.FFT_Mask_ForBack(batch_x, mask) # torch.Size([2, 1, 256, 256])
        optimizer.zero_grad()
        x_output, _ = model(batch_ipt)

        # Compute and print loss
        loss_discrepancy = torch.mean(torch.pow(x_output - noise, 2)) # = x - D_theta(x) = AWGN noise

        # loss_constraint = loss_fp_dist[layer_num-10]
        # for k in range(layer_num-10, layer_num-1):
        #     loss_constraint += loss_fp_dist[k+1]

        # gamma = torch.Tensor([0.000001]).to(device)

        # loss_all = loss_discrepancy
        loss_all = loss_discrepancy# + torch.mul(gamma, loss_constraint)
        # tensorboard
        snr  = compare_snr(x_output, batch_x)
        avg_loss_training = loss_all.item() + avg_loss_training
        avg_snr_training = snr.item() + avg_snr_training

        writer.add_scalar('train/snr_iter', snr.item(), global_step)
        writer.add_scalar('train/loss_all_iter', loss_all.item(), global_step)
        writer.add_scalar('train/loss_discrepancy_iter', loss_discrepancy.item(), global_step)
        # writer.add_scalar('train/loss_constraint_iter', loss_constraint.item(), global_step)

        # Zero gradients, perform a backward pass, and update the weights.
        
        loss_all.backward()

        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.01) 

        optimizer.step()

        batch = batch + 1
        global_step = global_step + 1

    with torch.no_grad():
        avg_snr_training = avg_snr_training / batch
        avg_loss_training = avg_loss_training / batch

        Img_pre = utils.make_grid(x_output, nrow=8, normalize=True, scale_each=True)
        Img_gdt = utils.make_grid(batch_x, nrow=8, normalize=True, scale_each=True)
        
        writer.add_image('train/pre', Img_pre, epoch_i, dataformats='CHW')
        writer.add_image('train/gt', Img_gdt, epoch_i, dataformats='CHW')

        writer.add_scalar('train/lr', current_lr, epoch_i)
        writer.add_scalar('train/snr_epoch', avg_snr_training, epoch_i)
        writer.add_scalar('train/loss_all_epoch', avg_loss_training, epoch_i)

    if epoch_i % 1 == 0:
        #####################################
        #               Valid               # 
        #####################################
        ImgNum = len(filepaths_test)

        test_gdt = np.zeros([ImgNum, 1, 256, 256], dtype=np.float32)
        test_pre = np.zeros([ImgNum, 1, 256, 256], dtype=np.float32)
        test_ipt = torch.zeros([ImgNum, 1, 256, 256])

        avg_psnr_testing = 0
        avg_ssim_testing = 0
        count = 0    

        model.eval()

        for batch_x in tqdm(valid_loader):

            Iorg_y = batch_x.detach().numpy()
            batch_x = batch_x.to(device)

            noise = torch.FloatTensor(batch_x.size()).normal_(mean=0, std=2/255.) # need to train (2,3,5)/255.
            batch_ipt = batch_x + noise

            # PhiTb = CSMRIClass.FFT_Mask_ForBack(batch_x, mask) # torch.Size([2, 1, 256, 256])
            
            [x_output, _] = model(batch_ipt, create_graph=False, strict=False)
            x_output = batch_ipt - x_output

            X_rec = x_output.detach().cpu().numpy()
            
            for ii in range(X_rec.shape[0]):
                rec_PSNR = psnr(X_rec[ii].astype(np.float64).squeeze(), Iorg_y[ii].astype(np.float64).squeeze())
                rec_SSIM = ssim(X_rec[ii].astype(np.float64).squeeze(), Iorg_y[ii].astype(np.float64).squeeze(), data_range=1)

                avg_psnr_testing = rec_PSNR + avg_psnr_testing
                avg_ssim_testing = rec_SSIM + avg_ssim_testing

                count = count + 1
        with torch.no_grad():
            avg_psnr_testing = avg_psnr_testing / count           
            avg_ssim_testing = avg_ssim_testing / count   

            writer.add_scalar('valid/psnr',avg_psnr_testing, epoch_i)
            writer.add_scalar('valid/ssim',avg_ssim_testing, epoch_i)
                    
            print("==================================\n")
            print("epoch: [%d]" % epoch_i)
            print('psnr: ', avg_psnr_testing)
            print("\n==================================")
            print("")

            Img_pre = utils.make_grid(x_output, nrow=3, normalize=True, scale_each=True)
            Img_gdt = utils.make_grid(batch_x, nrow=3, normalize=True, scale_each=True)
            # Img_ipt = utils.make_grid(PhiTb, nrow=3, normalize=True, scale_each=True)

            # Img_pre_iter = utils.make_grid(batch_pre_iter, nrow=3, normalize=True, scale_each=True)
            # writer.add_image('valid/ipt', Img_ipt, epoch_i, dataformats='CHW')
            writer.add_image('valid/pre', Img_pre, epoch_i, dataformats='CHW')
            writer.add_image('valid/gt', Img_gdt, epoch_i, dataformats='CHW')

            torch.save(model.state_dict(), '%s/net_params_%d.pkl' % (os.path.join(save_dir, 'log'), epoch_i))  # save only the parameters
            # torch.save(rObj.state_dict(), '%s/dnn_params_%d.pkl' % (os.path.join(save_dir, 'log'), epoch_i))  # save only the parameters

writer.close()