import torch
import numpy as np

x = [1,2,1,2,1,2]
tens = torch.tensor(x)
nump = np.array(x)

print("x is: ", x) # just a list
print("nump is: ", nump) #should be well known
print("tens is: ", tens) #very similar to numpy
print()
print(tens.numpy()) #convert to numpy
print(tens.shape) # get shape
print(tens.view(1,-1,1).shape) #can change dimensions
print(tens.view(1,-1,1)) # new dimensions
print()
print(tens.type()) # get type
print(tens.float().type()) # convert to float

