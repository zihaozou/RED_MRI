
from os import listdir, mkdir,chdir
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
import torch
from PIL.Image import open as imopen
from os.path import join
from model.jacob import jacobinNet
from model.cnn import DnCNN as spDnCNN
import matplotlib.pyplot as plt
imPath = 'DIV2K_valid_HR'
savePath = '/export1/project/DIV2K_PATCHED'
imageList = listdir(imPath)
testImage=np.asarray(imopen(join(imPath,imageList[0])))
patches=extract_patches_2d(testImage,(256,256),max_patches=5)
model=jacobinNet(spDnCNN(image_channels=3,pureCnn=True)).cuda()
pretrained = torch.load(
    '/export/project/jiaming.liu/Projects/Zihao/RED_MRI/logs/spec_5_multi_color_jacobian01:12:14/best.pt')
model.load_state_dict(pretrained['model'])
plt.imsave('test.png',patches[0,...])
x = torch.from_numpy(patches[0,...]).float().permute((2,0,1)).unsqueeze(0).cuda()/255.
noise = torch.FloatTensor(x.size()).normal_(mean=0, std=2./255.).cuda()
noisy=x+noise
noisy.requires_grad=True
predNoise=model(noisy)
recon=noisy-predNoise
from skimage.metrics import peak_signal_noise_ratio as psnr
print(psnr(x.cpu().detach().numpy(), recon.cpu().detach().numpy(),data_range=1))

