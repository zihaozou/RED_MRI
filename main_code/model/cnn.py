import torch.nn as nn
from utils.spectral_normalization import SpectralNorm


class DnCNN(nn.Module):
    def __init__(self, depth=12, n_channels=64, image_channels=1, kernel_size=3, pureCnn=False):
        super(DnCNN, self).__init__()
        padding = kernel_size // 2
        layers = []
        if pureCnn:
            l = nn.Conv2d(
                in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False)
        else:
            l = SpectralNorm(nn.Conv2d(
                in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(l)
        layers.append(nn.ReLU())

        for _ in range(depth-1):
            if pureCnn:
                l = nn.Conv2d(
                    in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False)
            else:
                l = SpectralNorm(nn.Conv2d(
                    in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(l)
            layers.append(nn.ReLU())
        if pureCnn:
            l = nn.Conv2d(
                in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=False)
        else:
            l = SpectralNorm(nn.Conv2d(
                in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(l)

        self.dncnn_3 = nn.Sequential(*layers)

    def forward(self, x_input):
        #return self.dncnn_3(x_input).mean([1, 2, 3]).sum()
        return self.dncnn_3(x_input)
