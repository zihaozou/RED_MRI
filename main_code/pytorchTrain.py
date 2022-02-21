from datetime import datetime
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
import PIL.Image as Image
from os.path import join
from os import listdir
import argparse
from tqdm import trange, tqdm
from torch.utils.tensorboard.writer import SummaryWriter
parser = argparse.ArgumentParser("CNN Trainer")
parser.add_argument('--no_jacob', dest='jacob', default=True, action='store_false',
                    help='jacobnet')
parser.add_argument('--conf_path', type=str,
                    help='config.json path')


class cnnTrainDataset(Dataset):
    def __init__(self, path, snr):
        dataFile = sio.loadmat(path)
        self.data = dataFile['labels']
        self.snr = snr

    def __getitem__(self, index):
        image = torch.Tensor(self.data[index, :]).float().unsqueeze(0)
        return image

    def __len__(self):
        return self.data.shape[0]


class cnnTestDataset(Dataset):
    def __init__(self, path, snr):
        self.dataPath = path
        self.snr = snr

    def __getitem__(self, index):
        image = torch.from_numpy(np.asarray(Image.open(
            join(self.dataPath, f"brain_test_{index+1:02d}.png")))).float().unsqueeze(0)/255.0
        return image

    def __len__(self):
        return len(listdir(self.dataPath))


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
    norm = config['cnn_model']['norm']
    pure = config['cnn_model']['pure']
    # training
    lr = config['train']['lr']
    weighDecay = config['train']['weigh_decay']
    batchSize = config['train']['batch_size']
    numTrain = config['train']['num_train']

    #create model
    jacob = jacobinNet(spDnCNN(depth=cnnDepth,
                               n_channels=cnnNumChans,
                               image_channels=cnnImageChans,
                               kernel_size=cnnKernelSize,
                               pureCnn=pure)).cuda()

    #create dataloader
    trainLoader = DataLoader(cnnTrainDataset(
        trainPath, SNR), batch_size=batchSize, pin_memory=True, num_workers=numWorkers, shuffle=True)
    valLoader = DataLoader(cnnTestDataset(valPath, SNR), batch_size=batchSize,
                           pin_memory=True, num_workers=numWorkers, shuffle=False)

    #create optimizer
    optimizer = torch.optim.Adam(jacob.parameters(),
                                 lr=lr, weight_decay=weighDecay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer, gamma=0.99)

    #create log writer
    run_name = args.conf_path.split('/')[-1].split('.')[0]
    if args.jacob:
        run_name += '_jacobian'
    run_name += datetime.now().strftime("%H:%M:%S")
    logger = SummaryWriter(log_dir=join(root_path, run_name),)
    for e in trange(numTrain):
        jacob.train()
        epochLoss=0
        for b, image in enumerate(tqdm(trainLoader)):
            optimizer.zero_grad()
            image = image.cuda()
            #noise = torch.FloatTensor(image.size()).normal_(mean=0, std=50./255.).cuda()
            noise = torch.zeros_like(image).cuda()
            noisyImage = image+noise
            noisyImage.requires_grad=True
            predNoise = jacob(noisyImage, create_graph=True, strict=True)
            loss = nn.functional.mse_loss(predNoise, noise)
            predArr=predNoise.detach().cpu().double().numpy()
            noiseArr=noise.detach().cpu().double().numpy()
            lossNP=np.mean(np.square(predArr-noiseArr))
            loss.backward()
            optimizer.step()
            scheduler.step()
            logger.add_scalar(tag='train_batch_loss',
                            scalar_value=loss.item(), global_step=b)
            epochLoss+=loss.item()
        logger.add_scalar(tag='train_loss',
                            scalar_value=epochLoss/len(trainLoader.dataset), global_step=e)
    torch.save(jacob.state_dict(),'test.pt')