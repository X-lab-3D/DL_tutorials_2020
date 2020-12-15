# -*- coding: utf-8 -*-

import torch
import imageio
import torch.nn as nn
import torch.nn.functional as F
import os

import matplotlib.pyplot as plt

from skimage import io, data, color
from skimage.transform import rescale, resize, downscale_local_mean

def dataset_resize():
    for fol in os.listdir('./'):
        if os.path.isdir(fol) and fol.endswith('_set'):
            i=0
            for img in os.listdir('./' + fol):
                if not img.startswith('img_'):
                    image = io.imread('./' + fol + '/' + img)
                    image_resized = resize(image, (100, 100),
                           anti_aliasing=True)
                    io.imsave('./' + fol + '/img_%i.png' %i, image_resized)
                    i+=1


def images2dataset(imagesLocation):
    final_tens = []
    for img in os.listdir(imagesLocation):
        if img.startswith('img_') and len(img.split('_')) == 2:
            tensor = torch.Tensor(imageio.imread(imagesLocation + img))/255
            tensor = tensor.permute(-1, 0, 1)[:3]
            final_tens.append(tensor)
    final_tens = torch.stack(final_tens)
    return final_tens

dataset_resize()

pos_set = images2dataset('pos_set/')
neg_set = images2dataset('neg_set/')
native_set = images2dataset('native_set/')
val_set = images2dataset('val_set/')
test_set = images2dataset('test_set/')

pos_laebls = torch.ones(20)
neg_laebls = torch.zeros(20)
#training_set = torch.cat((native_set, neg_set))
training_set = torch.cat((pos_set, neg_set))
labels = torch.cat((pos_laebls, neg_laebls))

net = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=18, kernel_size=1),
                    nn.BatchNorm2d(18),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    #nn.Softmax(),
                    nn.Conv2d(in_channels=18, out_channels=54, kernel_size=1),
                    nn.BatchNorm2d(54),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(in_channels=54, out_channels=1, kernel_size=1),
                    nn.Sigmoid())
optim = torch.optim.Adam(net.parameters(), lr=0.005)

epochs = 500
for e in range(epochs):
    loss = 0
    acc = 0
    prediction = net(training_set)
    means = prediction.mean(3).mean(2).squeeze()
    acc += (means.round() == labels).sum()/float(means.shape[0])
    lossF = F.binary_cross_entropy(means, labels)
    lossF.backward()
    optim.step()
    optim.zero_grad()
    loss += lossF.item()*training_set.size(0)
    if e % 20 == 0:
        #print('Epoch %i , Loss %.3f' %(e, loss))
        print('Epoch %i , Loss %.3f , Accuracy %.3f' %(e, loss, acc))
    #if acc == 1.000:
    #    break

net.eval()
validation = net(val_set)
test = net(test_set)

for i, image in enumerate(validation):
    imageio.imsave('val_output/%i.png' %i, validation[i].permute(1, 2, 0).detach())

