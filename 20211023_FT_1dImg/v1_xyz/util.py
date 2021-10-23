#!/usr/bin/env python
# Li Xue
#  2-Jan-2021 18:50

import torch
import pdb
from  math import pi as pi
import torchvision.transforms.functional as TF
import torch.nn as nn

from PIL import Image

class FT(nn.Module):
    # encode each element of x into sin(2*pi*f*n*x), cos(2*pi*f*n*x), where n = 1, 2, ..., N
    # f: fundamental frequency. f = 1/T, T is period, and is equal to the length of the integrable range of a signal
    # N: number of different frequencies
    # x: a vector or matrix. Each row a data point, each column a feature.
    # x size: [m,n], out size: [m, 2n]
    def __init__(self,f,N):
        super(FT, self).__init__()
        self.f = f
        self.N = N
    def forward(self,x):
        row = x.shape[0]
        col = x.shape[1]
        out = torch.tensor([])
        out = torch.zeros(row, col*2*self.N)

        j = 0
        for n in range(self.N):
            tmp1 = torch.sin(2*pi*self.f*(n+1)*x)
            tmp2 = torch.cos(2*pi*self.f*(n+1)*x)
            out[:,j*col:(j+1)*col]= tmp1
            out[:,(j+1)*col:(j+2)*col]= tmp2
            j = j + 2

        return out


class Img():
    def __init__(self, imgFL):
        self.FLname = imgFL

    def readImg(self):
        image = Image.open(self.FLname)
        image = TF.to_tensor(image)
        self.row = image.shape[1]
        self.col = image.shape[2]
        n_channel = image.shape[0] #RGB: 3
        self.shape = image.shape
        self.img = image
        self.n_channel = n_channel

        #flatten the image into 1D vector
        img_flat = image.view(n_channel,-1).permute(1,0)
        n_pixel = img_flat.shape[0]
        self.img_flat = img_flat
        self.n_pixel = n_pixel

        #get coordinates of the pixels - 1D
        self.coor_1D = torch.arange(0, n_pixel).float()
        self.coor_1D_norm = self.coor_1D/max(self.coor_1D)
        return self


