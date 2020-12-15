import torch
import torch.nn as nn

print("VECTOR CREATION")
# create a tensor
X = torch.tensor([[-1,-1.5,-1,-3,-1],
                  [0.1,-0.4,0.1,0.4,-0.5],
                  [2,1,3,3,1]]).float()
y = ['a', 'b', 'b'] # labels
X = X.view(3,-1) # 'Data'
Y = torch.tensor([0, 1, 1]) #labels as numbers (not one-hot encoded!)

print("X is:\n", X)
print("Y is:\n", Y)

print("\nLINEAR LAYER PART")
# create Linear layer
FC1 = nn.Linear(5,2) #5 features in, output of 2 (one per feature)
x = FC1(X) #apply linear layer
print("x after FC:\n",x) #weights are randomized
print("weights are: ", FC1._parameters)

print("\nRELU PART")
sig = nn.Sigmoid()
x = sig(x)
print("x after sigmoid:\n", x)

print("\nBATCHNORM PART")
bn = nn.BatchNorm1d(2)
x = bn(x)
print("x after batcnorm:\n", x)

print("\nDROPOUT PART")
drop = nn.Dropout(p=0.5)
x2 = drop(x)
print("x after dropout would be used:\n", x2)

print("\nSOFTMAX PART")
soft = nn.Softmax(dim=1)
x = soft(x)
print(x)

print("\nLOSS PART")
lossF = nn.CrossEntropyLoss()
loss = lossF(x2, Y)
print("the loss is:\n", loss)

print("\n(over)TRAIN THE LAYER")
optim = torch.optim.Adam(FC1.parameters(), lr = 0.001)
for i in range(500):
    optim.zero_grad()
    x = X.clone()
    x = FC1(x)
    x = soft(x)
    loss = lossF(x, Y)
    loss.backward()
    optim.step()
    if i %10 == 0:
        print("loss: ", loss)
        print(x)
print(FC1._parameters)

print("\n\nTRY OTHER DATA")

def throughLayer(x):
    x = FC1(x)
    x = soft(x)
    return x

x = torch.cat((torch.zeros(1,5), 2*torch.ones(1,5), -2*torch.ones(1,5), torch.tensor([-1,1,-1,1,0]).float().view(1,-1)))
print(throughLayer(x))
