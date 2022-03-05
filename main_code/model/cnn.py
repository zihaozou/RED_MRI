import torch.nn as nn
from utils.spectral_normalization import SpectralNorm
import math
import torch

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(
            2./9./64.)).clamp_(-0.025, 0.025)
        nn.init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight.data)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(
            2./9./64.)).clamp_(-0.025, 0.025)
        nn.init.constant_(m.bias.data, 0.0)
class DnCNN(nn.Module):
    def __init__(self, depth=12, n_channels=64, image_channels=1, kernel_size=3, pureCnn=False,bias=True):
        super(DnCNN, self).__init__()
        padding = kernel_size // 2

        layers = []
        if pureCnn:
            l = nn.Conv2d(
                in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        else:
            l = SpectralNorm(nn.Conv2d(
                in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=bias))
        layers.append(l)
        layers.append(nn.ELU())

        for _ in range(depth-1):
            if pureCnn:
                l = nn.Conv2d(
                    in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=bias)
            else:
                l = SpectralNorm(nn.Conv2d(
                    in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=bias))
            layers.append(l)
            layers.append(nn.ELU())
        if pureCnn:
            l = nn.Conv2d(
                in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        else:
            l = SpectralNorm(nn.Conv2d(
                in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=bias))
        layers.append(l)

        self.dncnn_3 = nn.Sequential(*layers)
        self.dncnn_3.apply(weights_init_kaiming)
    def forward(self, x_input):
        #return self.dncnn_3(x_input).mean([1, 2, 3]).sum()
        return self.dncnn_3(x_input)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            if m.bias is not None:
                m.bias.data.fill_(1)


class MultiDnCNN(nn.Module):
    def __init__(self, depth=12, n_channels=45, image_channels=1, kernel_size=[3,5,7], bias=True):
        super(MultiDnCNN, self).__init__()
        self.head = nn.Sequential(nn.Conv2d(
            in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size[0], padding=kernel_size[0]//2, bias=bias), nn.ELU())
        modules=[]
        for _ in range(depth-1):
            layers = []
            for k in kernel_size:
                l = nn.Sequential(nn.Conv2d(in_channels=n_channels, out_channels=n_channels//3,
                                            kernel_size=k, padding=k//2, bias=bias), nn.ELU())
                layers.append(l)
            layers=nn.ModuleList(layers)
            modules.append(layers)
        self.mods=nn.ModuleList(modules)
        self.tail =nn.Conv2d(
            in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size[0], padding=kernel_size[0]//2, bias=bias)
        self.mods.apply(weights_init_kaiming)
        self.tail.apply(weights_init_kaiming)
        self.head.apply(weights_init_kaiming)
    def forward(self, x_input):
        #return self.dncnn_3(x_input).mean([1, 2, 3]).sum()
        curr=self.head(x_input)
        for l in self.mods:
            currLst=[]
            for k in l:
                currLst.append(k(curr))
            curr=torch.cat(currLst,1)
        return self.tail(curr)
