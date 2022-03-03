from datetime import datetime
from genericpath import isdir
import numpy as np
import json
import torch
import torch.nn as nn
from model.cnn import DnCNN as spDnCNN
from model.jacob import jacobinNet
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
import numpy as np
from PIL.Image import open as imopen
from os.path import join
from os import listdir, mkdir, system
import argparse
from tqdm import trange, tqdm
from torch.utils.tensorboard.writer import SummaryWriter
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.feature_extraction.image import extract_patches_2d
import multiprocessing
from torch.nn import DataParallel as DP
from einops import rearrange
parser = argparse.ArgumentParser("CNN Trainer")
parser.add_argument('--no_jacob', dest='jacob', default=True, action='store_false',
                    help='jacobnet')
parser.add_argument('--conf_path', type=str,
                    help='config.json path')
parser.add_argument('--warm_up', type=str,default=None,
                    help='load pretrained state dict')

class cnnTrainDataset(Dataset):
    def __init__(self, path, snr):
        self.fileLst = listdir(path)
        self.snr = snr
        self.path = path

    def __getitem__(self, index):
        image = torch.load(join(self.path, self.fileLst[index]))
        return image

    def __len__(self):
        return len(self.fileLst)


class cnnTestDataset(Dataset):
    def __init__(self, path, snr):
        self.fileLst = listdir(path)
        self.snr = snr
        self.path = path

    def __getitem__(self, index):
        image = torch.load(join(self.path, self.fileLst[index]))
        return image

    def __len__(self):
        return len(self.fileLst)


def dataPatch(dataPath, fileLst, savePath,tv):
    for im in fileLst:
        imArr = np.asarray(imopen(join(dataPath, im)))
        patches = extract_patches_2d(imArr, (256, 256), max_patches=5)
        for i in range(patches.shape[0]):
            miniImg = torch.from_numpy(np.transpose(
                patches[i, ...], (2, 0, 1))).float()/255.
            name = im.split('.')[0]+'_'+str(i)+'.pt'
            torch.save(miniImg, join(savePath, tv, name))


def dataPreprocessMulti(trainPath, valPath, savePath,numProcess):
    if isdir(join(savePath, 'train')):
        system("rm %s -r" % (join(savePath, 'train')))
    if isdir(join(savePath, 'val')):
        system("rm %s -r" % (join(savePath, 'val')))
    mkdir(join(savePath, 'train'))
    trainLst=np.array_split(np.asarray(listdir(trainPath)),numProcess)
    jobs = []
    for i in range(numProcess):
        p = multiprocessing.Process(
            target=dataPatch, args=(trainPath, trainLst[i], savePath,'train',))
        jobs.append(p)
        p.start()
    for job in jobs:
        job.join()
    mkdir(join(savePath, 'val'))
    valLst = np.array_split(np.asarray(listdir(valPath)), numProcess)
    jobs = []
    for i in range(numProcess):
        p = multiprocessing.Process(
            target=dataPatch, args=(valPath, valLst[i], savePath, 'val',))
        jobs.append(p)
        p.start()
    for job in jobs:
        job.join()
    


def dataPostProcess(savePath):
    system("rm %s -r" % (join(savePath, 'train')))
    system("rm %s -r" % (join(savePath, 'val')))


if __name__ == '__main__':
    args = parser.parse_args()
    plt.ioff()
    with open(args.conf_path) as File:
        config = json.load(File)

    root_path = config['root_path']
    # GPU
    numGPU = config['GPUsetting']['num_gpus']
    GPUIndex = config['GPUsetting']['gpu_index']
    # data
    trainPath = config['dataset']['trainset']['path']
    valPath = config['dataset']['valset']['path']
    SNR = config['dataset']['snr']
    numWorkers = config['dataset']['num_workers']

    # model
    cnnDepth = config['cnn_model']['depth']
    cnnNumChans = config['cnn_model']['num_chans']
    cnnImageChans = config['cnn_model']['image_chans']
    cnnKernelSize = config['cnn_model']['kernel_size']
    pure = config['cnn_model']['pure']
    bias = config['cnn_model']['bias']
    # training
    lr = config['train']['lr']
    weighDecay = config['train']['weigh_decay']
    batchSize = config['train']['batch_size']
    numTrain = config['train']['num_train']
    showEvery=5
    device=config['GPUsetting']['gpu_index'][0]
    #test image
    testIm = torch.permute(torch.from_numpy(np.asarray(
        imopen(join('DIV2K_valid_HR/0801.png')))).float()/255., (2, 0, 1))
    # create model
    jacob = jacobinNet(spDnCNN(depth=cnnDepth,
                               n_channels=cnnNumChans,
                               image_channels=cnnImageChans,
                               kernel_size=cnnKernelSize,
                               pureCnn=pure, bias=bias)).cuda(device)
    if numGPU>1:
        jacob = DP(jacob, device_ids=GPUIndex)
    # create optimizer
    optimizer = torch.optim.Adam(jacob.parameters(),
                                 lr=lr, weight_decay=weighDecay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer, gamma=0.99)
    lossFunc = torch.nn.MSELoss()
    #
    if args.warm_up is not None:
        warmModel=torch.load(args.warm_up)
        jacob.load_state_dict(warmModel['model'])
        optimizer.load_state_dict(warmModel['optimizer'])
        scheduler.load_state_dict(warmModel['scheduler'])
    # create log writer
    run_name = args.conf_path.split('/')[-1].split('.')[0]+'_color'
    if args.jacob:
        run_name += '_jacobian'
    run_name += datetime.now().strftime("%H:%M:%S")
    logger = SummaryWriter(log_dir=join(root_path, run_name))
    mkdir(join('/export1/project/DIV2K_PATCHED', run_name))
    tempDataPath = join('/export1/project/DIV2K_PATCHED',run_name)
    bestModel = None
    bestPSNR = np.NINF
    dataPreprocessMulti(trainPath, valPath, tempDataPath, 10)
    for e in trange(numTrain):
        # create dataloader
        
        trainLoader = DataLoader(cnnTrainDataset(
            join(tempDataPath, 'train'), SNR), batch_size=batchSize, pin_memory=True, num_workers=numWorkers, shuffle=True)
        valLoader = DataLoader(cnnTestDataset(join(tempDataPath, 'val'), SNR), batch_size=batchSize,
                               pin_memory=True, num_workers=numWorkers, shuffle=False)
        jacob.train()
        epochLoss = 0
        # train
        for b,image in enumerate(pbar := tqdm(trainLoader)):
            optimizer.zero_grad()
            image = image.cuda(device)
            noise = torch.FloatTensor(image.size()).normal_(
                mean=0, std=SNR/255.).cuda(device)
            #noise = torch.zeros_like(image).cuda()
            noisyImage = image+noise
            noisyImage.requires_grad = True
            predNoise = jacob(noisyImage, create_graph=True, strict=True)
            loss = lossFunc(predNoise, noise)
            loss.backward()
            # grads_min = []
            # grads_max = []
            # for param in optimizer.param_groups[0]['params']:
            #     if param.grad is not None:
            #         grads_min.append(torch.min(param.grad))
            #         grads_max.append(torch.max(param.grad))

            # grads_min = torch.min(torch.stack(grads_min, 0))
            # grads_max = torch.max(torch.stack(grads_max, 0))
            optimizer.step()
            
            if b%showEvery==0:
                with torch.no_grad():
                    trainPSNR = psnr(image.detach().cpu().numpy(),
                                    (noisyImage-predNoise).detach().cpu().numpy(), data_range=1)
                    pbar.set_description("epoch %d, PSNR: %.2f,loss: %.5e" % (e,trainPSNR,loss.item()))
            epochLoss += loss.item()
        scheduler.step()
        logger.add_scalar(tag='train_loss',
                          scalar_value=epochLoss/len(trainLoader), global_step=e)
        jacob.eval()
        valPSNR = 0
        for image in valLoader:
            image = image.cuda(device)
            noise = torch.FloatTensor(image.size()).normal_(
                mean=0, std=SNR/255.).cuda(device)
            noisyImage = image+noise
            noisyImage.requires_grad = True
            predNoise = jacob(noisyImage, create_graph=False, strict=False)
            with torch.no_grad():
                valPSNR += psnr(image.detach().cpu().numpy(),
                                (noisyImage-predNoise).detach().cpu().numpy(), data_range=1)
        valPSNR /= len(valLoader)
        logger.add_scalar(tag='val_psnr',
                          scalar_value=valPSNR, global_step=e)
        if valPSNR > bestPSNR:
            bestModel = {'model': jacob.state_dict().copy(), 'optimizer': optimizer.state_dict(
            ).copy(), 'scheduler': scheduler.state_dict().copy()}
            bestPSNR = valPSNR
        
    dataPostProcess(tempDataPath)
    mostrecentModel={'model': jacob.state_dict().copy(), 'optimizer': optimizer.state_dict(
            ).copy(), 'scheduler': scheduler.state_dict().copy()
    }
    torch.save(bestModel, join(root_path, run_name, 'best.pt'))
    torch.save(mostrecentModel, join(root_path, run_name,'mostrecent.pt'))
