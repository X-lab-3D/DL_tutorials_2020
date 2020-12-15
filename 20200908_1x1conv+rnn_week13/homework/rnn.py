import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import random
import torch
        
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
    def forward(self, country_one_hot, input_one_hot, memory):
        return prediction, updated_memory
    def initMemory(self):
        return torch.zeros(1, self.hidden_size)
       
def letter2hot(letter):
	return oneHotVector
	
def name2hot(name):
	return oneHotTensor

def country2hot(country):
	return oneHotVector
	
def genNewName(country):
	return new_name

rnn = RNN(n_input, n_hidden)
memory = rnn.initHidden()
optim = torch.optim.Adam(rnn.parameters())
	
def train():
	pass

	
	
	
	

