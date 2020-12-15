import torch
import torch.nn as nn

class wineClassNet(nn.Module):
    def __init__(self):
        super(wineClassNet, self).__init__()

        self.FC1 = nn.Linear(12, 32)
        self.FC2 = nn.Linear(32, 64)
        self.FC3 = nn.Linear(64, 32)
        self.FC4 = nn.Linear(32,2)

        self.AF1 = nn.PReLU()
        self.AF2 = nn.PReLU()
        self.AF3 = nn.PReLU()
        self.AF4 = nn.Softmax(dim=1)

        self.BN1 = nn.BatchNorm1d(12)
        self.BN2 = nn.BatchNorm1d(32)
        self.BN3 = nn.BatchNorm1d(64)
        self.BN4 = nn.BatchNorm1d(32)

        self.drop = nn.Dropout(p=0.5)

    def forward(self, input):
        x = self.AF1(self.FC1(self.BN1(input)))
        x = self.AF2(self.FC2(self.BN2(x)))
        x = self.drop(self.BN3(x))
        x = self.AF3(self.FC3(x)) # it is already normalized before dropout
        x = self.AF4(self.FC4(self.BN4(x)))

        return x
