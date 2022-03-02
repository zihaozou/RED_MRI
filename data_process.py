from os import listdir, mkdir
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
import torch
from PIL.Image import open as imopen
from os.path import join
imPath = '/export1/project/Data_shirin/Data/DIV2K'
savePath = '/export1/project/DIV2K_PATCHED'
imageList = listdir(imPath)
for im in imageList:
    imArr = np.asarray(imopen(join(imPath, im)))
    patches = extract_patches_2d(imArr, (128, 128), max_patches=20)
    for i in range(patches.shape[0]):
        miniImg = torch.from_numpy(patches[i, ...]).float()
        name = im.split('.')[0]+'_'+str(i)+'.pt'
        torch.save(miniImg, join(savePath, name))
