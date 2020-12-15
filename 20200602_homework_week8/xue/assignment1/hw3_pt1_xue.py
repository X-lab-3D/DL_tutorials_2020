#!/usr/bin/env python
# Li Xue
# 27-Jun-2020 20:40

import sys
import numpy as np
import torch
import imageio
from matplotlib import pyplot as plt
from vae_example_conv import *
import pdb

img_ori = '../scripts_and_files_for_homework/dataset.png'

#--------------------------------------------------
#Assignment 1.1 data preprocessing

def image2dataset(image, outputshape):

    data_ori = imageio.imread(img_ori)
    num_emojis_row = int(data_ori.shape[0]/32)
    num_emojis_col = int(data_ori.shape[1]/32)
    num_emojis = num_emojis_row * num_emojis_col

    if outputshape == 'convolutional':
        data = torch.Tensor(num_emojis, 32, 32, 3)
        count = 0
        for i in range(num_emojis_row):
            for j in range(num_emojis_col):
                data[count] = torch.Tensor(data_ori[32*i:32*(i+1), 32*j:32*(j+1),:])
                count = count + 1

        data = data.permute([0,3,1,2])

    if outputshape == 'linear':
        data = torch.Tensor(num_emojis, 32 * 32 * 3)
        count = 0
        for i in range(num_emojis_row):
            for j in range(num_emojis_col):
                data[count] = torch.Tensor(data_ori[32*i:32*(i+1), 32*j:32*(j+1),:]).reshape(-1)
                count = count + 1
    data = data/255
    return count, data

num_img, data = image2dataset(img_ori, 'linear')
print(data.shape)

num_img, data = image2dataset(img_ori, 'convolutional')
#plt.imshow(data[1].permute([1,2,0]))
#plt.show()



#-------------------------------------------------
# Assignment 1.2, inner scientist: python gui.py

#-------------------------------------------------
# Assignment 1.3, image interplation

num_img, data = image2dataset(img_ori, 'convolutional')
def pick_2_imgs(data):
    idx = torch.randint(num_img, (1,2))
    idx1 = idx[0,0]
    idx2 = idx[0,1]
    print(f"Randomly choose two images: {idx1} and {idx2}")

def interpolate(x,y,prop = 0.5, method = 'latent'):

    prop = 0.5
    # interpolate on pixel space
    if method == 'pixel':
        xy = x*prop + y * (1-prop)

    # interpolate on laten space
    if method == 'latent':
        with torch.no_grad():
            net = VAE()
            net.eval()
            mu1, logvar1 = net.encode(x.unsqueeze(0))
            mu2, logvar2 = net.encode(y.unsqueeze(0))
            mu = mu1 * prop + mu2 * (1 - prop)
            xy = net.decode(mu).squeeze()
    return xy

def interpolate_range(x,y, steps = 10, method = 'latent'):
    newImgs = torch.Tensor(steps+2, 3, 32, 32 )
    newImgs[0] = y # attach the starting img

    #-- start interpolating
    for i in range(steps):
        prop = (i+1)*0.1
        xy = interpolate(x,y,prop, method )
        pdb.set_trace()
        newImgs[i+1] = xy

    newImgs[steps+1] = x # attach the target img
    print(newImgs.shape)
    newImgs = (newImgs *255).permute([0, 2, 3, 1 ]).reshape((steps+2)*32, 32,3).to(torch.uint8).numpy()
    print(newImgs.shape)
    return newImgs


idx1 = 0
idx2 = 3
x = data[idx1]
y = data[idx2]
#newImgs_pxl = interpolate_range(x,y, steps = 10 , method = 'pixel')
newImgs_lat = interpolate_range(x,y, steps = 1 , method = 'latent')
#imageio.imsave(f"{idx1}_{idx2}_pxl.png",newImgs_pxl)
imageio.imsave(f"{idx1}_{idx2}_lat.png",newImgs_lat)
print(f"images interpolated and save as png files")
sys.exit()

#-------------------------------------------------
# Assignment 1.4, math with laten vectors

a = imageio.imread('../scripts_and_files_for_homework/a.png')
b = imageio.imread('../scripts_and_files_for_homework/b.png')
c = imageio.imread('../scripts_and_files_for_homework/c.png')
d = imageio.imread('../scripts_and_files_for_homework/d.png')
e = imageio.imread('../scripts_and_files_for_homework/e.png')

data = torch.Tensor(5,32,32,3)
data[0] = torch.Tensor(a)
data[1] = torch.Tensor(b)
data[2] = torch.Tensor(c)
data[3] = torch.Tensor(d)
data[4] = torch.Tensor(e)
data = data.permute([0,3,1,2])
net = VAE()
checkpoint = torch.load('../scripts_and_files_for_homework/checkpoint.pth.tar')
net.load_state_dict(checkpoint['state_dict'])
net.eval()
img_reconstr, mu, logvar = net(data/255)
img_reconstr = img_reconstr*255

z = net.reparameterize(mu, logvar)
img = net.decode(z) * 255
img = img.permute([0,2,3,1]).to(torch.uint8).numpy()
plt.imshow(img[0])
plt.show()

# a - b + c
mu_new = mu[0] - mu[1] + mu[2]
logvar_new = logvar[0] - logvar[1] + logvar[2]
z_new = net.reparameterize(mu_new, logvar_new)
torch.save(z_new, 'a_b+c.pt')

img_new = net.decode(z_new.unsqueeze(0) ) * 255
img_new = img_new[0].permute([1,2,0]).to(torch.uint8).numpy()
plt.imshow(img_new)
plt.show()

# a - b + d
mu_new = mu[0] - mu[1] + mu[3]
logvar_new = logvar[0] - logvar[1] + logvar[3]
z_new = net.reparameterize(mu_new, logvar_new)
torch.save(z_new, 'a_b+d.pt')

img_new = net.decode(z_new.unsqueeze(0) ) * 255
img_new = img_new[0].permute([1,2,0]).to(torch.uint8).numpy()
plt.imshow(img_new)
plt.show()

# a - b + e
mu_new = mu[0] - mu[1] + mu[4]
logvar_new = logvar[0] - logvar[1] + logvar[4]
z_new = net.reparameterize(mu_new, logvar_new)
torch.save(z_new, 'a_b+e.pt')

img_new = net.decode(z_new.unsqueeze(0) ) * 255
img_new = img_new[0].permute([1,2,0]).to(torch.uint8).numpy()
plt.imshow(img_new)
plt.show()

#-------------------------------------------------
# Assignment 1.5, anomaly detection

num_img, data = image2dataset(img_ori, 'convolutional')
net = VAE()
net.eval()
checkpoint = torch.load('../scripts_and_files_for_homework/checkpoint.pth.tar')
net.load_state_dict(checkpoint['state_dict'])

losses= torch.Tensor(num_img)
for i in range(num_img):
    img = data[i]/255
    img = img.unsqueeze(0)
    img_reconstr, mu, logvar = net(img)
    loss = specialLoss(img_reconstr, img, mu, logvar)
    losses[i] = loss

#-- top 10 worst reconstructions
values, indices = torch.topk(losses,5)
badImgs = data[indices].permute([0,2,3,1]).to(torch.uint8).numpy()
badImgs_reconstr, mu, logvar = net(data[indices]/255)
badImgs_reconstr = (badImgs_reconstr *255).permute([0,2,3,1]).to(torch.uint8).numpy()
plt.imshow(badImgs[0])
plt.show()
plt.imshow(badImgs_reconstr[0])
plt.show()


