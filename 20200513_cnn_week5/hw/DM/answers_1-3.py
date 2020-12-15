#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 14:31:58 2020

@author: dario
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input feature, 8 output features, kernelsize of 5 x 5 pixels
        self.conv1 = nn.Conv2d(1, 8, 5, padding=1) 
        # Pool operation 2 x 2 
        self.pool = nn.MaxPool2d(2, 2)
        # Norm
        self.norm1 = nn.BatchNorm1d(8)
        self.norm2 = nn.BatchNorm2d(8)
        self.norm3 = nn.BatchNorm3d(8)
        

x=torch.randn(8,1,5,5)
net=Net()

print([(q,z) for q, z in zip(F.relu(net.pool(net.conv1(x))), net.pool(F.relu(net.conv1(x))))])
#print(x)

### Question 1:

#There is no difference because torch.rand is only providing positive values.
#There is no difference when using randn as well because at least one value per each element is positive, 
#so that value will be returned from maxpooling and the flattening given by relu is not perceptible.


#%%
# Define a convolutional neural network
Net = nn.Sequential(nn.Conv2d(1, 8, 5),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(8, 16, 5),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                    nn.Flatten(),
                    nn.Linear(16 * 4 * 4, 256),
                    nn.ReLU(),
                    nn.Linear(256, 64),
                    nn.ReLU(),
                    nn.Linear(64, 10),
                    nn.Softmax(dim=1))



#y = torch.rand(8,1,12,12)
x = torch.rand(8,1,28,28)
net=Net(x)
print(net)

### This is not working. Why?


#%%

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input feature, 8 output features, kernelsize of 5 x 5 pixels
        self.conv1 = nn.Conv2d(1, 8, 5) 
        self.norm2d = nn.BatchNorm2d(8)
        self.norm1d = nn.BatchNorm1d(256)
        # Pool operation 2 x 2 
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)
    def forward(self, x):
    	# conv -> activation function -> pooling
        x = self.pool(F.relu(self.conv1(x)))
        # conv -> activation function -> pooling
        x = self.norm2d(x)                                             ### Question 2: Here BatchNorm2 since there is a 4D input
        x = self.pool(F.relu(self.conv2(x)))
        # Transition to fully connected layers
        x = x.view(-1, 16 * 4 * 4)
        x = self.norm1d(x)                                            ### Question 2: Here BatchNorm1 since there is a 3D input
        # Normal linear layers with activation functions
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # Softmax for probability distribution over 10 classes
        return F.softmax(x, dim=1) 
    
x = torch.rand(256,1,16,16)
net=Net()
print(net(x))

### Question 3a:
# Because we are convoluting with a kernel of size 5, thus the output is smaller. It is 6 because a 5x5 filter has to 
# slide on 6 different positions per side to cover the whole 10x10 matrix

### Question 3b:
# No

### Question 3c:
# It an additional, fake layer arount our matrix that will prevent us from loosing information on the borders
# and allow us to get an output with the same shape of the input

