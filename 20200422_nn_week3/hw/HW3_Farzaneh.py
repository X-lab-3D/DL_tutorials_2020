# -*- coding: utf-8 -*-


#Problem  1
'''
The error is raised because we have a BatchNorm1d layer. If we omit this Batch layer it works without error. The reason is that with single sample the model expects more then 1 value to calculate the running mean and std of the current batch.

To avoid this we can add neuralnet.eval() before feeding with the input. As this will change the behavior of the BatchNorm layer to use the running estimates instead of calculating them for the current batch
'''
#Problem 2
'''
This occurs because it expects 784 features not 28 x 28 features.
Simply reshape of the input by .view(32, 784) it works or use.view (-1).
'''


import torch
import torch.nn as nn
import torch.functional as F


#Probelm 3 -A
class MySigmoid(nn.Module):
      def __init__(self):
              super(MyModule , self).__init__()
      def forward(self, x):
              return 1/(1+torch.exp(-x))
                      
#Problem 3 -B
class MySoftmax(nn.Module):
        def __init__(self,dim):
             super(MySoftmax , self).__init__()
             self.dim = dim
        def forward(self, x):
             maxes = torch.max(x, self.dim , keepdim=True)[0]
             x_exp = torch.exp(x-maxes)
             x_exp_sum = torch.sum(x_exp, self.dim , keepdim=True)
             return x_exp/x_exp_sum     
            
#x = torch.Tensor([[2,1,0.1],[2,1,0.1]])
#dim = 0
#y=torch.zeros(x.shape)
#
#for i in range(x.shape[dim]):
#     sum_val = torch.sum(torch.exp(x[i]))
#     y[i] = torch.exp(x[i])/sum_val
#     print(y[i])

#Test
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,6,3)
        self.conv2 = nn.Conv2d(6,16,3)        
         
        self.fc1 = nn.Linear(16*6*6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self,x):
        x= F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x= F.max_pool2d(F.relu(self.conv2(x)), 2)
        x= x.view(-1, self.num_flat_features(x))
        x= F.relu(self.fc1(x))
        x= F.relu(self.fc2(x))
        x= self.fc3(x)
        return x
    def num_flat_features(self,x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()    


#Problem 4 - AND
neuralnet = nn.Sequential( nn.Linear(2,50),
       nn.ReLU(),
       nn.BatchNorm1d(50),
       nn.Linear(50,1),
       nn.Sigmoid())

def calcAccuracy(predictions, labels):
	return (torch.round(predictions)==labels).sum()/float(len(predictions)) * 100

optim = torch.optim.SGD(neuralnet.parameters(), lr= 0.001)
data = torch.Tensor([[0,0],[0,1],[1,0],[1,1]])
labels = torch.Tensor([0,0,0,1]).view(4,1)
for epoch in range(1000):
       prediction =  neuralnet(data)
       #print(prediction)
       train_Acc  = calcAccuracy(prediction,  labels)
       loss = torch.nn.functional.mse_loss(prediction, labels)
       loss.backward()
       optim.step()
       optim.zero_grad()  
       if epoch % 50 == 0:
           print('Epoch %i. train acc: %.2f%% train loss: %.2f%%'% (epoch, train_Acc,loss))
           

#Problem 4 - OR
neuralnet = nn.Sequential( nn.Linear(2,50),
       nn.ReLU(),
       nn.BatchNorm1d(50),
       nn.Linear(50,1),
       nn.Sigmoid())

def calcAccuracy(predictions, labels):
	return (torch.round(predictions)==labels).sum()/float(len(predictions)) * 100

optim = torch.optim.SGD(neuralnet.parameters(), lr= 0.001)
data = torch.Tensor([[0,0],[0,1],[1,0],[1,1]])
labels = torch.Tensor([0,1,1,1]).view(4,1)
for epoch in range(1000):
       prediction =  neuralnet(data)
       #print(prediction)
       train_Acc  = calcAccuracy(prediction,  labels)
       loss = torch.nn.functional.mse_loss(prediction, labels)
       loss.backward()
       optim.step()
       optim.zero_grad()  
       if epoch % 50 == 0:
           print('Epoch %i. train acc: %.2f%% train loss: %.2f%%'% (epoch, train_Acc,loss))     
        
#Problem 4 - XOR
neuralnet = nn.Sequential( nn.Linear(2,50),
       nn.ReLU(),
       nn.BatchNorm1d(50),
       nn.Linear(50,1),
       nn.Sigmoid())

def calcAccuracy(predictions, labels):
	return (torch.round(predictions)==labels).sum()/float(len(predictions)) * 100

optim = torch.optim.SGD(neuralnet.parameters(), lr= 0.001)
data = torch.Tensor([[0,0],[0,1],[1,0],[1,1]])
labels = torch.Tensor([0,1,1,0]).view(4,1)
for epoch in range(1000):
       prediction =  neuralnet(data)
       #print(prediction)
       train_Acc  = calcAccuracy(prediction,  labels)
       loss = torch.nn.functional.mse_loss(prediction, labels)
       loss.backward()
       optim.step()
       optim.zero_grad()  
       if epoch % 50 == 0:
           print('Epoch %i. train acc: %.2f%% train loss: %.2f%%'% (epoch, train_Acc,loss))

              
# Problem 5:
a = torch.ones(3,3)
b = torch.zeros(3,3)

c = torch.cat((a.view(1,3,3),b.view(1,3,3)), dim=0)
