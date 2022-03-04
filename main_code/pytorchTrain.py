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
from torchvision.transforms.functional import crop
import matplotlib.pyplot as plt
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

class dataPreparer(object):
    def __init__(self, trainPath, valPath, savePath, numProcess,sizeMin,sizeMax) -> None:
        self.trainPath=trainPath
        self.valPath=valPath
        self.savePath=savePath
        self.numP=numProcess
        self.tempFolderName=['0','1']
        self.sizeRange=(sizeMin,sizeMax)
        if isdir(join(self.savePath, self.tempFolderName[0])):
            system("rm %s -r" % (join(self.savePath, self.tempFolderName[0])))
        mkdir(join(self.savePath, self.tempFolderName[0]))
        if isdir(join(self.savePath, self.tempFolderName[1])):
            system("rm %s -r" % (join(self.savePath, self.tempFolderName[1])))
        mkdir(join(self.savePath, self.tempFolderName[1]))
        self.curr=0
    def start(self):
        self.data_runner()
    def contin(self):
        folder=self.getCurrFolder
        pid = multiprocessing.Process(target=self.data_runner)
        pid.start()
        self.pid=pid
        return folder
    def wait(self):
        self.pid.join()
    def cleanUp(self):
        if isdir(join(self.savePath, self.tempFolderName[0])):
            system("rm %s -r" % (join(self.savePath, self.tempFolderName[0])))
        if isdir(join(self.savePath, self.tempFolderName[1])):
            system("rm %s -r" % (join(self.savePath, self.tempFolderName[1])))
    
    @property
    def getNextFolder(self):
        self.curr=(self.curr+1)%2
        return join(self.savePath,self.tempFolderName[self.curr])

    @property
    def getCurrFolder(self):
        return join(self.savePath, self.tempFolderName[self.curr])

    def data_runner(self):
        size = np.random.randint(*self.sizeRange)
        currfolder = self.getNextFolder
        if isdir(join(currfolder,'train')):
            system("rm %s -r" % (join(currfolder, 'train')))
        if isdir(join(self.savePath, currfolder, 'val')):
            system("rm %s -r" % (join(currfolder, 'val')))
        mkdir(join(currfolder, 'train'))
        trainLst = np.array_split(np.asarray(listdir(self.trainPath)), self.numP)
        jobs = []
        for i in range(self.numP):
            p = multiprocessing.Process(
                target=self.dataPatch, args=(self.trainPath, trainLst[i], join(currfolder, 'train'),size,))
            jobs.append(p)
            p.start()
        for job in jobs:
            job.join()
        mkdir(join(currfolder, 'val'))
        valLst = np.array_split(np.asarray(listdir(self.valPath)), self.numP)
        jobs = []
        for i in range(self.numP):
            p = multiprocessing.Process(
                target=self.dataPatch, args=(self.valPath, valLst[i], join(currfolder, 'val'),size,))
            jobs.append(p)
            p.start()
        for job in jobs:
            job.join()

    def dataPatch(self,dataPath, fileLst, savePath,size):
        for im in fileLst:
            imArr = np.asarray(imopen(join(dataPath, im)))
            patches = extract_patches_2d(imArr, (size,size), max_patches=3)
            for i in range(patches.shape[0]):
                miniImg = torch.from_numpy(np.transpose(
                    patches[i, ...], (2, 0, 1))).float()/255.
                name = im.split('.')[0]+'_'+str(i)+'.pt'
                torch.save(miniImg, join(savePath, name))

def randomCrop(img,size):
    assert(size<=img.shape[2])
    assert(size<=img.shape[3])
    x=np.random.randint(low=0,high=img.shape[2]-size)
    y = np.random.randint(low=0, high=img.shape[3]-size)
    return crop(img,x,y,size,size).detach()

def preview(img,model,writer,snr,device,e):
    cropLst=[256,512,1024]
    fig, axs = plt.subplots(nrows=3, ncols=2, constrained_layout=True)
    for i,s in enumerate(cropLst):
        cropped=randomCrop(img,s)
        croppedN =(cropped+ torch.FloatTensor(cropped.size()).normal_(
            mean=0, std=snr/255.)).detach()
        croppedN.requires_grad_()
        predNoise=model(croppedN.to(device),False,False).detach().cpu()
        recon=torch.clamp(croppedN-predNoise,min=0,max=1).detach()
        testPSNR=psnr(cropped.numpy(),recon.numpy(),data_range=1)
        noisyPSNR = psnr(cropped.numpy(), torch.clamp(
            croppedN, min=0, max=1).detach().numpy(), data_range=1)
        axs[i,0].imshow(recon.squeeze().permute((1,2,0)))
        axs[i,1].imshow(cropped.squeeze().permute((1,2,0)))
        axs[i,0].text(2,30,f'{testPSNR:.2f}')
        axs[i, 1].text(2, 30, f'{noisyPSNR:.2f}')
    writer.add_figure('preview',fig,e)
    plt.close(fig)
        
        





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
        imopen(join('DIV2K_valid_HR/0801.png')))).float()/255., (2, 0, 1)).unsqueeze(0)
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

    dataPre=dataPreparer(trainPath,valPath,tempDataPath,5,64,512)
    dataPre.start()
    for e in trange(numTrain):
        # create dataloader
        folder=dataPre.contin()
        trainLoader = DataLoader(cnnTrainDataset(
            join(folder, 'train'), SNR), batch_size=batchSize, pin_memory=True, num_workers=numWorkers, shuffle=True,drop_last=True)
        valLoader = DataLoader(cnnTestDataset(join(folder, 'val'), SNR), batch_size=batchSize,
                               pin_memory=True, num_workers=numWorkers, shuffle=False, drop_last=True)
        jacob.train()
        epochLoss = 0
        #train
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
        preview(testIm,jacob,logger,SNR,device,e)
        dataPre.wait()
    dataPre.cleanUp()
    mostrecentModel={'model': jacob.state_dict().copy(), 'optimizer': optimizer.state_dict(
            ).copy(), 'scheduler': scheduler.state_dict().copy()
    }
    torch.save(bestModel, join(root_path, run_name, 'best.pt'))
    torch.save(mostrecentModel, join(root_path, run_name,'mostrecent.pt'))
