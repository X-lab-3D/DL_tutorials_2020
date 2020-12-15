#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 12:30:08 2020

@author: dario
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

#%%
neuralnet = nn.Sequential( nn.Linear(784, 500),
                           nn.ReLU(),
                           nn.BatchNorm1d(500),
                           nn.Linear(500, 10),
                           nn.Softmax()
                           )

test1 = torch.rand(1, 784)
test2 = torch.rand(2, 784)

neuralnet(test2)
'''
Answer 1:
    neuralnet needs at least two samples to apply the batch normalization between them.
    With test1 we only provide one sample.
'''

test3 = torch.rand(32, 28, 28)
test4 = test3.view(32, 784)      #Answer 2 or test3.view(32, -1)
neuralnet(test4)

#%%
class MySigModule(nn.Module):
    def __init__(self):
        super(MySigModule, self).__init__()
    def forward(self, x):
        y = 1.0 / (1.0 + torch.exp(-x))
        return y


neuralnet = nn.Sequential( nn.Linear(784, 500),
                           nn.ReLU(),
                           nn.BatchNorm1d(500),
                           nn.Linear(500, 10),
                           MySigModule()
                           )
neuralnet(test2)

class MySoftModule(nn.Module):
    def __init__(self, p1):
        super(MySoftModule, self).__init__()
        self.p1 = p1
    def forward(self, x):
        div = sum(torch.exp(x[self.p1]))
        return torch.exp(y)/div

#%%
#Question 4

### AND
print('AND FUNCTION')
net = nn.Sequential(nn.Linear(2, 100),
                    nn.ReLU(),
                    nn.BatchNorm1d(100),
                    nn.Linear(100,1),
                    nn.Sigmoid())
optim = torch.optim.Adam(net.parameters(), lr=0.001)

AND_train_set = ((1, (1,1)), (0, (0,1)), (0, (1,0)), (0, (0,0)))
data = torch.Tensor([x[1] for x in AND_train_set])
labels = torch.Tensor([x[0] for x in AND_train_set]).view(-1,1)

epochs = 300
for e in range(epochs):
    loss = 0
    acc = 0
    prediction = net(data)
    acc += (prediction.round() == labels).sum()/float(prediction.shape[0])
    lossF = F.mse_loss(prediction, labels)
    lossF.backward()
    optim.step()
    optim.zero_grad()
    loss += lossF.item()*data.size(0)
    if e % 20 == 0:
        print('Epoch %i , Loss %.3f , Accuracy %.3f' %(e, loss, acc))


#%%
### OR
print('OR FUNCTION')
net = nn.Sequential(nn.Linear(2, 100),
                    nn.ReLU(),
                    nn.BatchNorm1d(100),
                    nn.Linear(100,1),
                    nn.Sigmoid())
optim = torch.optim.Adam(net.parameters(), lr=0.001)

OR_train_set = ((1, (1,1)), (1, (0,1)), (1, (1,0)), (0, (0,0)))
data = torch.Tensor([x[1] for x in OR_train_set])
labels = torch.Tensor([x[0] for x in OR_train_set]).view(-1,1)

epochs = 300
for e in range(epochs):
    loss = 0
    acc = 0
    prediction = net(data)
    acc += (prediction.round() == labels).sum()/float(prediction.shape[0])
    lossF = F.mse_loss(prediction, labels)
    lossF.backward()
    optim.step()
    optim.zero_grad()
    loss += lossF.item()*data.size(0)
    if e % 20 == 0:
        print('Epoch %i , Loss %.3f , Accuracy %.3f' %(e, loss, acc))

#%%
### XOR
print('XOR FUNCTION')
net = nn.Sequential(nn.Linear(2, 100),
                    nn.ReLU(),
                    nn.BatchNorm1d(100),
                    nn.Linear(100,1),
                    nn.Sigmoid())
optim = torch.optim.Adam(net.parameters(), lr=0.001)

XOR_train_set = ((0, (1,1)), (1, (0,1)), (1, (1,0)), (0, (0,0)))
data = torch.Tensor([x[1] for x in XOR_train_set])
labels = torch.Tensor([x[0] for x in XOR_train_set]).view(-1,1)

epochs = 300
for e in range(epochs):
    loss = 0
    acc = 0
    prediction = net(data)
    acc += (prediction.round() == labels).sum()/float(prediction.shape[0])
    lossF = F.mse_loss(prediction, labels)
    lossF.backward()
    optim.step()
    optim.zero_grad()
    loss += lossF.item()*data.size(0)
    if e % 20 == 0:
        print('Epoch %i , Loss %.3f , Accuracy %.3f' %(e, loss, acc))

#%%
#Question 5
        
x = torch.ones(3,3)
y = torch.ones_like(x)
z = torch.cat((x, y), 0) # 6x3
m = torch.cat((x, y), 1) # 3x6
p = torch.stack((x, y), 2) # 2x3x3