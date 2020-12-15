# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from Bio import SeqIO

#########################
####### VARIABLES #######

global nt_dict
global aa_dict

nt_dict = {'A': [1.,0.,0.,0.], 'T': [0.,1.,0.,0.],
           'G': [0.,0.,1.,0.], 'C': [0.,0.,0.,1.]}

aa_dict = {'G' : [], 'S' : [], 'T' : [], 'N' : [], 'C' : [], 'M' : [], 'Q' : [], 'D' : [], 'E' : [], 'K' : [],
           'H' : [],'R' : [], 'I' : [], 'W' : [], 'Y' : [], 'P' : [], 'A' : [], 'F' : [], 'L' : [], 'V' : [], 
           '*' : []}

#fill aa_dict
for i, key in enumerate(aa_dict):
    for j in range(len(aa_dict)):
        if j == i:
            aa_dict[key].append(1.)
        else:
            aa_dict[key].append(0.)


#compile reverse dicts
rev_nt_dict = {}
rev_aa_dict = {}

for i, key in enumerate(nt_dict):
    rev_nt_dict[i] = key
    
for i, key in enumerate(aa_dict):
    rev_aa_dict[i] = key

#########################
####### FUNCTIONS #######

def seq_encoder(seq, kind=None):
    if kind == 'nt':
        encoder = nt_dict
        length = 4
    elif kind == 'aa':
        encoder = aa_dict
        length = 21
    out_vector = []
    for letter in seq:
        tensor = torch.tensor(encoder[letter])
        out_vector.append(tensor.view(1,length))
    return(torch.stack(tuple(out_vector), -1))

def prob_dist_decoder(tensor, kind=None):
    if kind == 'nt':
        length = 4
    elif kind == 'aa':
        length = 21
    out_tens = []
    for i in range(tensor.size(-1)):
        zeros = torch.zeros(1,length)
        indices = torch.argmax(tensor[:, :, i])
        indx = indices.tolist()
        zeros[0][indx] = 1.0
        out_tensor = zeros
        out_tens.append(out_tensor)
    return(torch.stack(tuple(out_tens), -1))

def seq_decoder(tensor, kind=None):
    if kind == 'nt':
        decoder = rev_nt_dict
    elif kind == 'aa':
        decoder = rev_aa_dict
    out_seq = ''
    for index in range(tensor.size(-1)):
        out_tensor = tensor[:, :, index]
        for i, bit in enumerate(out_tensor.tolist()[0]):
            if bit == 1.0:
                out_seq += decoder[i]
    return out_seq

#########################
####### NEURALNET #######

net = nn.Sequential(nn.Conv1d(4, 16, 3, stride=3),
                    nn.BatchNorm1d(16),
                    nn.ReLU(),
                    nn.Conv1d(16, 10, 1),
                    nn.BatchNorm1d(10),
                    nn.ReLU(),
                    nn.Conv1d(10,21, 1),
                    nn.Softmax(dim=1))

optim = torch.optim.Adam(net.parameters())

#########################
########## RUN ##########
DNA = []
prot = []
with open('./seqs.fasta', 'r') as infile:
    records = list(SeqIO.parse(infile, "fasta"))
for i, x in enumerate(records):
    if i%2:
        prot.append(str(x.seq))
    else:
        DNA.append(str(x.seq))
        

totalLoss = 0.0
for epoch in range(1000):
    losses = []
    for dna_seq, prot_seq in zip(DNA, prot):
        optim.zero_grad()
        nt_tensor = seq_encoder(dna_seq, 'nt')
        aa_tensor = seq_encoder(prot_seq, 'aa')
        net.eval()
        prediction = net(nt_tensor)
        pr = prediction.view(1, 21, len(prot_seq)) # I have no idea how the net could already output the correct sized tensor
        dpr = prob_dist_decoder(pr, 'aa')
        #loss_f = nn.BCELoss()
        loss = F.binary_cross_entropy(prediction, aa_tensor, reduction='sum')
        #loss = nn.BCELoss()(dpr, aa_tensor)
        #loss = loss_f(dpr, aa_tensor)
        loss.backward()
        optim.step()
        totalLoss += float(loss)
        losses.append(float(loss))
        if epoch == 999:
            print(seq_decoder(aa_tensor, 'aa'))
            print(seq_decoder(dpr, 'aa'))
    if epoch%100 == 0:
        print(epoch, losses)
        
    
#Question 4e:
# B and D, because they have a wrong number of features. All the others are just very short sequences
