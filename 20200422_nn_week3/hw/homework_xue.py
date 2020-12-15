#!/usr/bin/env python
# Li Xue
#  2-May-2020 15:51

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import numpy as np
import time
import pdb


#QN 1: BatchNorm does not work with one data (no standard deviation can be calculated)
#QN 2:
a = torch.rand(32,28,28)
a.view(32,-1)
a.reshape(32,-1)

#QN3 a:
class my_act1(nn.Module):
    def __init__(self):
        super(my_act1, self).__init__()
    def forward (self, x):
        return  torch.sigmoid(x)

new_act1 = my_act1()
data = torch.rand(2,3)
new_act1(data)

#QN3 b:
class my_act2(nn.Module):
    def __init__(self,dim):
        super(my_act2, self).__init__()
        self.dim = dim
    def forward(self,x):
        return F.softmax(x,self.dim)
new_act2 = my_act2(dim=0)
data=torch.rand(2,3)
data
new_act2(data)
torch.sum(new_act2(data),dim = 0)

#QN4:

def gen_binaryTensor(size):
    a = torch.rand(size)
    a[a>0.5] = 1
    a[a<=0.5] = 0
    return a

def gen_ANDdata(num_data):
    x = gen_binaryTensor([num_data,2])
    y = torch.zeros(num_data, 1)
    tmp = torch.sum(x, dim = 1)
    y[tmp==2] = 1
    return x,y


def gen_ORdata(num_data):
    x = gen_binaryTensor([num_data,2])
    y= torch.zeros(num_data, 1)
    tmp = torch.sum(x, dim = 1)
    y[tmp>=1] = 1
    return x, y

def gen_XORdata(num_data):
    x = gen_binaryTensor([num_data,2])
    y = torch.zeros(num_data, 1)
    tmp = torch.sum(x, dim = 1)
    y[tmp == 1] = 1
    return x, y

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(2,100)
        self.bn1 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100,100)
        self.bn2 = nn.BatchNorm1d(100)
        self.fc3 = nn.Linear(100,1)
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        y = torch.sigmoid(self.fc3(x))

        return y

net = Net()
#print(net)
#summary(net, input_size=(1,1,2)) # input_size=(channels, H, W)

def train(x,y, net, epoches = 1, batch_size=4):
    t0 = time.time()
    #-- train
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    losses = []
    epoch = 0

    while epoch < epoches:

        #-- shuffle training set
        randomIdx = torch.randperm(x.shape[0])
        x = x[randomIdx]
        y = y[randomIdx]

        #-- train
        batch_size = batch_size
        num_batches = int(np.floor(x.shape[0]/batch_size))
        for i in range(0, num_batches):
            minibatch_x = x[i*batch_size:(i+1)* batch_size ]
            minibatch_y = y[i*batch_size:(i+1)* batch_size]

            optimizer.zero_grad()
            out = net(minibatch_x)
            criterion = nn.MSELoss()
            loss = criterion(out, minibatch_y )
            loss.backward()
            optimizer.step()

            if i % np.floor(num_batches/5) ==0:
                print(f"epoch = {epoch}, batch = {i}, loss = {loss:.4f}")
                losses.append(loss)
        epoch+= 1

    print(f' --> Training done in {time.time()-t0:.2f} sec')
    return net, losses

def evaluate(net):
    #check results
    net.eval()
    x = torch.Tensor([[0,0],[0,1],[1,0],[1,1]])
    probs = net(x) #predicted probability
    y_pred = torch.round(probs)

    for i in range(x.shape[0]):
        print(f"x={x[i]}, y_pred={y_pred[i].item()}, prob = {probs[i].item():.2f}")


#-- 1. AND
print ("\n=======> AND\n")
x,y = gen_ANDdata(200)
net = Net()
net,losses = train(x,y, net)
evaluate(net)

#-- 2. OR
print ("\n=======> OR\n")
x,y = gen_ORdata(200)
net = Net()
net,losses = train(x,y, net)
evaluate(net)

#-- 3. XOR
print ("\n=======> XOR\n")
x,y = gen_XORdata(200)
net = Net()
net,losses = train(x,y, net)
evaluate(net)


# QN5:
a = torch.ones(3,3)
b = torch.zeros(3,3)

torch.cat( (a,b), 0 )
torch.cat( (a,b), 1 )
torch.stack((a,b))
