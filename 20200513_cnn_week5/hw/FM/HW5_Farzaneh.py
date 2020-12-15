#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 02:44:54 2020

@author: farzanehmeimandi
"""

# Problem 1
# As we are using Max-pooling, it does not differ in the order of Max-pooling and the activation function

# Problem 2
# 2d & 1d (in the following code) - It seems to be a debate whether using BatchNormalization before or after activation
# Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
import torch
import torch.nn as nn
import torch.nn.functional as F
from Bio import SeqIO
'''
class Net(nn.Module):
      def __init__(self)	:
          super(Net , self).__init__()
          self.conv1 = nn.Conv2d(1,8,5)
          self.pool = nn.MaxPool2d(2,2)
          self.bn2d = nn.BatchNorm2d(8)
          self.conv2 = nn.Conv2d(8, 16, 5)
          self.bn2d = nn.BatchNorm2d(16)
          self.fc1 = nn.Linear(16 * 4 * 4, 256)
          self.bn1d = nn.BatchNorm1d(256)
          self.fc2 = nn.Linear(256, 64)
          self.bn1d = nn.BatchNorm1d(64)
          self.fc3 = nn.Linear(64, 10)
      def forward(self,x):
          x = self.pool(F.relu(self.conv1(x)))
          x = self.bn2d(x)
          x = self.pool(F.relu(self.conv2(x)))
          x = self.bn2d(x)
          x = x.view(-1, 16 * 4 * 4)
          x = F.relu(self.fc1(x))
          x = self.bn1d(x)
          x = F.relu(self.fc2(x))
          x = self.bn1d(x)
          x = self.fc3(x)
          return F.softmax(x,dim=1)

Net = nn.Sequential(nn.Conv2d(1,8,5),
                    nn.ReLU(),
                    nn.MaxPool2d(2,2),
                    nn.BatchNorm2d(8),
                    nn.Conv2d(8,16,5),
                    nn.ReLU(),
                    nn.MaxPool2d(2,2),
                    nn.BatchNorm2d(16),
                    nn.Flatten(),
                    nn.Linear(16 * 4 * 4, 256),
                    nn.ReLU(),
                    nn.BatchNorm1d(256),
                    nn.Linear(256, 64),
                    nn.ReLU(),
                    nn.BatchNorm1d(64),
                    nn.Linear(64, 10),
                    nn.Softmax(dim=1)
                    )     
''' 
# Problem 3a 
# With convolution sliding a window through the data, the edges are not covered. 
# If we have N x N image size and F x F filter size then after convolution result will be
# (N x N) * (F x F) = (N-F+1)x(N-F+1)  --> For our case : (10-5+1) x (10-5+1) = 6 x 6
conv =  nn.Conv2d(1,1,5)
input = torch.rand(1,1,10,10)
output = conv(input)

          
# Problem 3b --> no it does not differ using kernel-size 1
          
# Problem 3c
# To maintain the dimension of output as in input , we use padding. Padding is a process of adding zeros to the input matrix symmetrically to allow for more space for the kernel to cover the image.      
# https://towardsdatascience.com/covolutional-neural-network-cb0883dd6529

# Problem 4

net = nn.Sequential(
        nn.Conv1d(4, 10 , 3,  stride=3),
        nn.BatchNorm1d(10),
        nn.ReLU(),
        nn.Conv1d(10, 20, 1),
        nn.BatchNorm1d(20), 
        nn.ReLU(), 
        nn.Conv1d(20,21, 1), 
        nn.Softmax(dim=1))
        

# One hot encoding
NAs = torch.eye(4, dtype=torch.int)
AAs = torch.eye(21, dtype=torch.int)


def decode2AA(enocded_res_seq):
    AA_codes= ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y','*']
    decoded_seq = []
    for i in range(len(enocded_res_seq)):
        decoded_seq.append(AA_codes[enocded_res_seq[i].argmax(0)])
    decoded_seq=''.join(decoded_seq)
    return decoded_seq

def encodeAA(prot_seq):
    prot_dict = {'A':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7, 'I':8, 'K':9, 'L':10, 'M':11, 'N':12, 'P':13, 'Q':14, 'R':15, 'S':16, 'T':17, 'V':18, 'W':19, 'Y':20,'*':21}
    i=0
    encoded_prot_seq=torch.zeros(len(prot_seq),len(prot_dict))
    for k in prot_seq:
       encoded_prot_seq[i,:]=AAs[prot_dict[k]-1]
       i+=1
    return encoded_prot_seq
    
def encodeNA(DNA_seq):
    NA_dict = {'A': 1, 'T':2 , 'C':3, 'G':4}
    i=0
    encoded_DNA_seq=torch.zeros(len(DNA_seq),len(NA_dict))
    for k in DNA_seq:
       encoded_DNA_seq[i,:]=NAs[NA_dict[k]-1]
       i+=1
    return encoded_DNA_seq

def calcAccuracy(prediction, labels, reduce=True):
	overlap = (prediction.argmax(1)==labels.argmax(1)).sum()
	if reduce:
		return overlap/float(labels.size(2))
	return overlap

optim = torch.optim.Adam(net.parameters()) # the optimizer for the gradients

train_sequences = SeqIO.parse(open('sequences.fasta'),'fasta')
DNAs=[]
proteins=[]
for record in SeqIO.parse("sequences.fasta", "fasta"):
    if ("DNA" in record.description):
        DNAs.append(str(record.seq))
    elif ("PROTEIN" in record.description):
        proteins.append(str(record.seq))

for epoch in range(int(400)):
    for DNA, prot in zip(DNAs, proteins):
        optim.zero_grad()
        DNA_train=encodeNA(DNA)
        labels_train=encodeAA(prot).T.unsqueeze(0)
        prediction=net(DNA_train.T.unsqueeze(0))
        net.eval()
        loss = F.binary_cross_entropy(prediction, labels_train)
        loss.backward() # Calculate gradients
        optim.step() # Update gradients
        loss = float(loss) 
        accuracy  = calcAccuracy(prediction, labels_train, True)
        if ((epoch+1)%10)==0:
            print(decode2AA(prediction.squeeze(0).T))
            print(prot)
    
    if ((epoch+1)%10)==0:
        print('\nEpoch: %i\t loss: %.4f\t accuracy %.2f%%' % (epoch+1,loss, accuracy*100))


# Problem 4c:
#Stride is the number of pixels a convolutional filter moves, like a sliding window  
#https://avvineed.com/convolutional-neural-network-numpy.html  
    
# Problem 4d:
    
# PRoblem 4e: B and D and F
    