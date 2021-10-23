#!/usr/bin/env python
# Li Xue
#  1-Jan-2021 14:51

import torch
from Net import *
import torchvision
import torch.optim as optim
from tqdm import tqdm
import pdb
import matplotlib
from matplotlib import pyplot as plt
from torch.autograd import gradcheck
from util import *

matplotlib.use('Agg')

#read img and prepare input and output data
img = Img('cat.png').readImg()
input = img.coor_1D.unsqueeze(1)  # 0, 1, ...., 120*120
rgb = img.img_flat #rgb.shape = [n_pixel, 4]

#network param
criterion = nn.MSELoss()
net = Net()
opt = optim.Adam(params=net.parameters())

#start training
losses = []

for epoch in tqdm(torch.arange(0,3500)):

    idx = torch.randperm(img.n_pixel)
    net.zero_grad()
    out = net(input[idx,])
    #out = net(input)
    loss = criterion(out, rgb[idx,])
    #loss = criterion(out, rgb)
    loss.backward()
    losses.append(loss.item())
    opt.step()

    if epoch % 500 ==0:
        print(f"epo = {epoch}, loss = {loss:.5f}")
        print(f"target = {rgb[0:2,:]}")
        print(f"out = {out[0:2,:]}")
        # restore the order of pixels
        out_tmp = [pred_rgb for _, pred_rgb in sorted(zip(idx,out))]
        out = torch.stack(out_tmp)
        # 1D to 2D
        img_pred = out.permute(1,0).reshape(img.n_channel, img.row, img.col)
        torchvision.utils.save_image(img_pred, f'pred_{epoch}.png')

plt.plot(losses)
plt.savefig('loss.png')
