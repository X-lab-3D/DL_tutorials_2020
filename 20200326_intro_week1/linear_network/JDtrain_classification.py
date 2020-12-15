from JDload_data import makeWineSamples
from JDlinearNets import wineClassNet

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

# load data
wines_train, wines_test, labels_train, labels_test, wines_eval, labels_eval = makeWineSamples()

# set some parameters
NR_TRAIN = len(wines_train)
NR_EVAL = len(wines_eval)
NR_SAMPLES = 500
PRINT_every = 10
SAVE_every = 25

# define loss functions, create net and add some
lossF = nn.CrossEntropyLoss()
net = wineClassNet()
runCounter = 0
lowerCounter = 0
lowest = 99999999999
losses = []

#create an optimizer
optimizer = torch.optim.Adam(net.parameters(), lr = 0.0001)

goOn = True
printLoss = torch.tensor(0.0)
evalLoss = torch.tensor(0.0)

while(goOn):
    optimizer.zero_grad()
    samples = np.random.random_integers(0,NR_TRAIN-1, size=NR_SAMPLES)
    x = wines_train[samples].view(NR_SAMPLES, -1)
    y = labels_train[samples]

    output = net(x)
    loss = lossF(output,y)
    printLoss += loss

    loss.backward()
    optimizer.step()

    if runCounter % PRINT_every == 0:
        print(printLoss/PRINT_every)
        losses += [float(printLoss / PRINT_every)]
        printLoss = 0

    if runCounter % SAVE_every == 0:
        evalSamples = np.random.random_integers(0, NR_EVAL - 1, size=NR_SAMPLES * 5)
        x = wines_eval[evalSamples].view(NR_SAMPLES * 5, -1)
        y = labels_eval[evalSamples]
        output = net(x)
        thisLoss = lossF(output, y)
        if thisLoss < lowest:
            lowest = thisLoss
            print("eval Loss is:", thisLoss)
            lowerCounter = 0
        else:
            if lowerCounter >= 5:
                goOn = False
            else:
                lowerCounter +=1

    runCounter +=1

# get metrics on test-set
testVals = net(wines_test)
testVals, testIDX = testVals.max(1)
print("\nThe confusion matrix of the testSet")
print(confusion_matrix(labels_test, testIDX))

plt.plot(losses[1:])
plt.ylabel("loss")
plt.show()
