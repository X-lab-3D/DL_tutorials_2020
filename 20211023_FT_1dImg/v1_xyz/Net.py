#!/usr/bin/env python
# Li Xue
#  1-Jan-2021 16:11

import torch
import torch.nn as nn
import torch.nn.functional as F
from util import *


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(1,50*3*2)
        self.bn1 = nn.BatchNorm1d(50*3*2)

        self.fc2 = nn.Linear(50*3*2,50)
        self.bn2 = nn.BatchNorm1d(50)

        self.fc3 = nn.Linear(50,50)
        self.bn3 = nn.BatchNorm1d(50)

        self.fc4 = nn.Linear(50,4)

    def forward(self, x):
        x = F.elu(self.bn1(self.fc1(x)))
        x = F.elu(self.bn2(self.fc2(x)))
        x = F.elu(self.bn3(self.fc3(x)))
        x = torch.sigmoid(self.fc4(x))
        return x
