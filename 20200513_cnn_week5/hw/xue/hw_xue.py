#!/usr/bin/env python
# Li Xue
# 23-May-2020 15:56

import sys
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import pandas as pd
import numpy as np
import time
import pdb
from Bio import SeqIO

#--QN1
X = torch.rand(1,1,5,5)
conv = nn.Conv2d(1,1,3, padding = 1)
pool = nn.MaxPool2d(2, 2)
output1 = pool(F.relu(conv(X)))
output2 = F.relu(pool(conv(X)))
output1
output2

#--QN4
amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
            'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '*']
nuclotides = ['A', 'T', 'C', 'G']
def codebook(seq_type = 'protein'):
    # generate a feature lookup dataframe for proteins or DNA

    if seq_type  == 'protein':
        names =  amino_acids
    elif seq_type == 'DNA':
        names = nuclotides
    code = np.eye(len(names), dtype=int)
    code = pd.DataFrame(code)
    code.columns = names
    return code

code_prot = codebook('protein')
code_DNA = codebook('DNA')

def one_hot_encode(seq, seq_type='protein'):
    # encode a protein/DNA seq into one-hot
    if seq_type == 'protein':
        x = pd.DataFrame([code_prot[i] for i in seq])
    elif seq_type == 'DNA':
        x = pd.DataFrame([code_DNA[i] for i in seq])
    return x

def df_2_tensor(df):
   # input (df):
   #        0  1  2  3
   #     A  1  0  0  0
   #     T  0  1  0  0
   X = torch.Tensor(df.values)
   X = X.permute(1,0)
   X = X.unsqueeze(0)
   return X

def translate(mat):
    # input:  a torch.tensor with size of 21 x L, L is the length of a protein
    # output: a protein seq
    idx = torch.argmax(mat, dim = 0) # get the class ID
    seq = [amino_acids[i] for i in idx]
    seq = ''.join(seq)
    return seq


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv1d(in_channels = 4, out_channels = 8, kernel_size= 3, stride = 3)
        self.bn1 = nn.BatchNorm1d(8)
        self.conv2 = nn.Conv1d(8,16, kernel_size = 1)
        self.bn2 = nn.BatchNorm1d(16)
        self.conv3 = nn.Conv1d(16, 21, kernel_size = 1)
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        y = self.conv3(x)
        y = F.softmax(self.conv3(x), dim = 1)
        return y

model = Net()

def train(x, y, epoches = 1000):
    t0 = time.time()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    epoch = 0
    while epoch < epoches:
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        epoch += 1

        if epoch % 50 == 0 :
            print(f"epoch = {epoch}, loss = {loss:.4f}")
    return out

DNAs = SeqIO.parse(open('DNAs.fasta'),'fasta')
prots = SeqIO.parse(open('proteins.fasta'),'fasta')

for DNA, prot in zip(DNAs, prots):
    print (f"\n----> Train on {DNA.id} and {prot.id}")
    X = one_hot_encode(DNA.seq, seq_type = 'DNA')
    Y = one_hot_encode(prot.seq, seq_type = 'protein')
    X = df_2_tensor(X)
    Y = df_2_tensor(Y)
    Y = torch.argmax(Y, dim = 1) # get the class ID
    out = train(X,Y) # out.shape = [1, 21, 102]
    prot_pred = translate(torch.squeeze(out))
    print(prot_pred)
    print(prot.seq)

#----------------------------------------------
#-- 4e. test on the unseen codons for Proline: cct, ccc, cca, ccg
print(f"\nTest on the unseen codons: cct, ccc, cca, ccg")
torch.save(model, './trained.model')
model = torch.load('./trained.model')
model.eval()
X = one_hot_encode('cctcccccaccg'.upper(), seq_type = 'DNA')
X = df_2_tensor(X)
Y = model(X)
prot_pred = translate(torch.squeeze(Y))
print(prot_pred)


#----------------------------------------------
#-- 4f. check acceptable input tensor shapes
# Tensor A of shape (1, 4, 3): works
# Tensor B of shape (1, 5, 9): No. The number of input channels should be 4 not 5.
# Tensor C of shape (1, 4, 6): works
# Tensor D of shape (1, 1, 7): No. The number of input channels should be 4 not 1.
# Tensor E of shape (1, 4, 7): works.
# Tensor F of shape (1, 4, 2): No, too short (2 condons). Should be at least 3 condons (kernel size = 3)

conv = torch.nn.Conv1d(4,8,3, stride = 3)
conv(torch.rand(1,4,3)

