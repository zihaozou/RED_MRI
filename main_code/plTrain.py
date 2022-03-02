from datetime import datetime
import json
from tabnanny import verbose
from typing import Any
import pytorch_lightning as pl
import torch
import torch.nn as nn
from model.models import DnCNN
from model.cnn import DnCNN as spDnCNN
from model.jacob import jacobinNet
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from utils.util import compare_snr
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
import numpy as np
import PIL.Image as Image
from os.path import join
from os import listdir
from pytorch_lightning.callbacks import ModelCheckpoint
from matplotlib.gridspec import GridSpec
import argparse
from model.gspnp.network_unet import UNetRes
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
        return image, torch.FloatTensor(image.size()).normal_(mean=0, std=self.snr/255.)

    def __len__(self):
        return self.data.shape[0]


class cnnTestDataset(Dataset):
    def __init__(self, path, snr):
        self.dataPath = path
        self.snr = snr

    def __getitem__(self, index):
        image = torch.from_numpy(np.asarray(Image.open(
            join(self.dataPath, f"brain_test_{index+1:02d}.png")))).float().unsqueeze(0)/255.0
        return image, torch.FloatTensor(image.size()).normal_(mean=0, std=self.snr/255.)

    def __len__(self):
        return len(listdir(self.dataPath))


class plWrapper(pl.LightningModule):
    def __init__(self, cnnDepth=12,
                 cnnNumChannels=64,
                 cnnImageChannels=1,
                 cnnKernelSize=3,
                 trainingPath='Dataset/BrainImages_train/Training_BrainImages_256x256_100.mat',
                 valPath='Dataset/BrainImages_test',
                 previewImagePath='Dataset/BrainImages_test/brain_test_01.png',
                 snr=2,
                 lr=5e-4,
                 weighDecay=1e-8,
                 norm='spec',
                 jacob=True,
                 ** kwargs: Any):
        super().__init__()
        self.save_hyperparameters()
        if self.hparams['type'] == 'conv':
            self.dncnn = spDnCNN(depth=self.hparams['cnnDepth'],
                                 n_channels=self.hparams['cnnNumChannels'],
                                 image_channels=self.hparams['cnnImageChannels'],
                                 kernel_size=self.hparams['cnnKernelSize'],
                                 pureCnn=self.hparams.pure, bias=self.hparams['bias'])
        elif self.hparams.type == 'unet':
            self.dncnn = UNetRes(in_nc=1, out_nc=1, act_mode='E',nb=self.hparams['numBlock'],bias=self.hparams['bias'])
        if self.hparams['jacob']:
            self.jacob = jacobinNet(self.dncnn)

        previewImage = torch.from_numpy(np.asarray(
            Image.open(previewImagePath))).float().unsqueeze(0).unsqueeze(0)/255.0
        previewNoise = torch.FloatTensor(
            previewImage.size()).normal_(mean=0, std=snr/255.)
        self.register_buffer(name='previewImage', tensor=previewImage)
        self.register_buffer(name='previewNoise', tensor=previewNoise)

    def forward(self, X, create_graph=True, strict=True):
        if self.hparams['jacob']:
            return self.jacob(X, create_graph, strict)
        else:
            return self.dncnn(X)

    def training_step(self, batch, batch_idx):
        image, noise = batch
        noisyImage = image+noise
        predNoise = self(noisyImage, create_graph=True, strict=True)

        loss = torch.mean(torch.pow(predNoise-noise, 2))
        with torch.no_grad():
            train_snr = compare_snr(image, noisyImage-predNoise)
        self.log("train_loss", loss, on_step=False,
                 on_epoch=True, prog_bar=False, logger=True)
        self.log("train_snr", train_snr, on_step=False,
                 on_epoch=True, prog_bar=False, logger=True)
        return {'loss': loss, 'snr': train_snr}

    def on_validation_model_eval(self, *args, **kwargs):
        super().on_validation_model_eval(*args, **kwargs)
        torch.set_grad_enabled(True)

    def validation_step(self, batch, batch_idx):
        image, noise = batch
        noisyImage = image+noise
        predNoise = self(noisyImage, create_graph=False, strict=False)
        #predNoise = self(noisyImage)
        valPSNR = 0
        valSSIM = 0
        with torch.no_grad():
            for i in range(image.shape[0]):
                valPSNR += peak_signal_noise_ratio(
                    image[i, 0, :, :].detach().cpu().numpy(), torch.clamp(noisyImage[i, 0, :, :]-predNoise[i, 0, :, :], min=0, max=1).detach().cpu().numpy(), data_range=1)
                valSSIM += structural_similarity(image[i, 0, :, :].detach().cpu(
                ).numpy(), torch.clamp(noisyImage[i, 0, :, :]-predNoise[i, 0, :, :], min=0, max=1).detach().cpu().numpy(), data_range=1)
        self.log("val_psnr", valPSNR/image.shape[0], on_step=False,
                 on_epoch=True, prog_bar=False, logger=True)
        self.log("val_ssim", valSSIM/image.shape[0], on_step=False,
                 on_epoch=True, prog_bar=False, logger=True)
        return {'psnr': valPSNR/image.shape[0], 'ssim': valSSIM/image.shape[0]}

    def validation_epoch_end(self, outputs) -> None:
        previewNoisyImage = self.previewImage+self.previewNoise
        predPreviewNoise = self(
            previewNoisyImage, create_graph=False, strict=False)
        prevPSNR = peak_signal_noise_ratio(
            self.previewImage.detach().cpu().numpy().squeeze(), torch.clamp(previewNoisyImage-predPreviewNoise, min=0, max=1).detach().cpu().numpy().squeeze(), data_range=1)
        prevSSIM = structural_similarity(self.previewImage.detach().cpu(
        ).numpy().squeeze(), torch.clamp(previewNoisyImage-predPreviewNoise, min=0, max=1).detach().cpu().numpy().squeeze(), data_range=1)
        noisyPSNR = peak_signal_noise_ratio(
            self.previewImage.detach().cpu().numpy().squeeze(), previewNoisyImage.detach().cpu().numpy().squeeze(), data_range=1)
        noisySSIM = structural_similarity(self.previewImage.detach().cpu(
        ).numpy().squeeze(), previewNoisyImage.detach().cpu().numpy().squeeze(), data_range=1)
        fig = plt.figure(constrained_layout=True, figsize=(10, 2.6), dpi=100)

        subfigs = fig.subfigures(1, 3, wspace=0)
        gs0 = GridSpec(nrows=2, ncols=3,
                       figure=subfigs[0], wspace=0, hspace=0)

        main0 = subfigs[0].add_subplot(gs0[:, 0:2])
        sub00 = subfigs[0].add_subplot(gs0[0, 2])
        sub01 = subfigs[0].add_subplot(gs0[1, 2])
        gs1 = GridSpec(nrows=2, ncols=3,
                       figure=subfigs[1], wspace=0, hspace=0)

        main1 = subfigs[1].add_subplot(gs1[:, 0:2])
        sub10 = subfigs[1].add_subplot(gs1[0, 2])
        sub11 = subfigs[1].add_subplot(gs1[1, 2])
        gs2 = GridSpec(nrows=2, ncols=3,
                       figure=subfigs[2], wspace=0, hspace=0)

        main2 = subfigs[2].add_subplot(gs2[:, 0:2])
        sub20 = subfigs[2].add_subplot(gs2[0, 2])
        sub21 = subfigs[2].add_subplot(gs2[1, 2])
        for sf in subfigs:
            for ax in sf.axes:
                ax.set_axis_off()
        subfigs[0].suptitle('Original Image')
        main0.imshow(self.previewImage.detach().cpu().squeeze(),
                     cmap='gray')
        sub00.imshow(self.previewImage.detach().cpu().squeeze()
                     [42:77, 75:110], cmap='gray')
        sub01.imshow(self.previewImage.detach().cpu().squeeze()
                     [160:195, 136:171],  cmap='gray')
        subfigs[1].suptitle('Noisy Image')
        main1.imshow(previewNoisyImage.detach().cpu().squeeze(),
                     cmap='gray')
        main1.text(
            0, 240, f'PSNR: {noisyPSNR:.2f}dB; SSIM: {noisySSIM:.2f}', color='white', fontsize='small')
        sub10.imshow(previewNoisyImage.detach().cpu().squeeze()
                     [42:77, 75:110], cmap='gray')
        sub11.imshow(previewNoisyImage.detach().cpu().squeeze()
                     [160:195, 136:171],  cmap='gray')
        subfigs[2].suptitle('Denoised Image')
        main2.imshow(torch.clamp(previewNoisyImage-predPreviewNoise,
                     min=0, max=1).detach().cpu().squeeze(), cmap='gray')
        main2.text(
            0, 240, f'PSNR: {prevPSNR:.2f}dB; SSIM: {prevSSIM:.2f}', color='white', fontsize='small')
        sub20.imshow(torch.clamp(previewNoisyImage-predPreviewNoise,
                     min=0, max=1).detach().cpu().squeeze()
                     [42:77, 75:110], cmap='gray')
        sub21.imshow(torch.clamp(previewNoisyImage-predPreviewNoise,
                     min=0, max=1).detach().cpu().squeeze()
                     [160:195, 136:171],  cmap='gray')
        self.logger.experiment.add_figure(
            'preview', fig, self.current_epoch+1)

    def train_dataloader(self):
        num_workers = self.hparams['num_workers'] if \
            'num_workers' in self.hparams.keys() else 1
        batch_size = self.hparams['batch_size'] if 'batch_size' in self.hparams.keys(
        ) else 1
        return DataLoader(cnnTrainDataset(self.hparams['trainingPath'],
                                          snr=self.hparams['snr']),
                          batch_size=batch_size,
                          shuffle=True,
                          pin_memory=True,
                          num_workers=num_workers)

    def val_dataloader(self):
        num_workers = self.hparams['num_workers'] if \
            'num_workers' in self.hparams.keys() else 1
        batch_size = self.hparams['batch_size'] if 'batch_size' in self.hparams.keys(
        ) else 1
        return DataLoader(cnnTestDataset(self.hparams['valPath'],
                                         snr=self.hparams['snr']),
                          batch_size=batch_size,
                          shuffle=False,
                          pin_memory=True,
                          num_workers=num_workers)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(
        ), lr=self.hparams['lr'], weight_decay=self.hparams['weighDecay'])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer, gamma=0.99)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}


if __name__ == "__main__":
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
    cnntype = config['cnn_model']['type']
    pure = config['cnn_model']['pure']
    bias=config['cnn_model']['bias']
    # training
    lr = config['train']['lr']
    weighDecay = config['train']['weigh_decay']
    batchSize = config['train']['batch_size']
    numTrain = config['train']['num_train']
    # create model
    model = plWrapper(cnnDepth=cnnDepth,
                      cnnNumChannels=cnnNumChans,
                      cnnImageChannels=cnnImageChans,
                      cnnKernelSize=cnnKernelSize,
                      trainingPath=trainPath,
                      valPath=valPath,
                      snr=SNR,
                      lr=lr,
                      weighDecay=weighDecay,
                      num_workers=numWorkers,
                      batch_size=batchSize,
                      type=cnntype,
                      jacob=args.jacob,
                      pure=pure,
                      bias=bias)
    # create trainer
    run_name = args.conf_path.split('/')[-1].split('.')[0]
    if args.jacob:
        run_name += '_jacobian'
    run_name += datetime.now().strftime("%H:%M:%S")
    ckptCallback = ModelCheckpoint(
        join(root_path, run_name, 'best_model'),
        save_top_k=1,
        monitor='val_psnr',
        mode='max',
        filename='best_model')
    trainer = pl.Trainer(default_root_dir=join(root_path, run_name),
                         gpus=numGPU,
                         max_epochs=numTrain,
                         num_sanity_val_steps=0,
                         check_val_every_n_epoch=1,
                         enable_checkpointing=True,
                         callbacks=[ckptCallback],
                         gradient_clip_val=0.01,
                         gradient_clip_algorithm='value')
    trainer.fit(model)
