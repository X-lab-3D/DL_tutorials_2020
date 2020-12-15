import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

'''Simple convolutional neural network example'''

# Download mnist train data
mnist  = torchvision.datasets.MNIST('.', train=True, download=True)
imagesTrain = mnist.data/255. # -> images, normalized between 0 and 1
imagesTrain = imagesTrain.unsqueeze(1) # Add the 'feature dimension/axis'
labelsTrain = mnist.targets   # -> the labels, 0-9
labelsTrain = torch.eye(10)[labelsTrain] # -> quickly transform to oneHot vectors

# Download mnist test data
mnist  = torchvision.datasets.MNIST('.', train=False, download=True)
imagesTest = mnist.data/255. # -> images, normalized between 0 and 1
imagesTest = imagesTest.unsqueeze(1) # Add the 'feature dimension/axis'
labelsTest = mnist.targets   # -> the labels, 0-9
labelsTest = torch.eye(10)[labelsTest] # -> quickly transform to oneHot vectors

# Define a convolutional neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input feature, 8 output features, kernelsize of 5 x 5 pixels
        self.conv1 = nn.Conv2d(1, 8, 5) 
        # Pool operation 2 x 2 
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)
    def forward(self, x):
    	# conv -> activation function -> pooling
        x = self.pool(F.relu(self.conv1(x)))
        # conv -> activation function -> pooling
        x = self.pool(F.relu(self.conv2(x)))
        # Transition to fully connected layers
        x = x.view(-1, 16 * 4 * 4)
        # Normal linear layers with activation functions
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # Softmax for probability distribution over 10 classes
        return F.softmax(x, dim=1) 

def calcAccuracy(prediction, labels, reduce=True):
	overlap = (prediction.argmax(1)==labels.argmax(1)).sum()
	if reduce:
		return overlap/float(len(labels))
	return overlap

def calculateTestLoss(imagesTest, labelsTest):
	with torch.no_grad(): # -> No need to calculate gradients on test data
		prediction = net(imagesTest)
		loss = F.binary_cross_entropy(prediction, labelsTest,  reduction='sum')
		accuracy  = calcAccuracy(prediction, labelsTest)
	# Return the average loss and accuracy
	return float(loss)/len(imagesTest), accuracy

def trainOneEpoch(imagesTrain, labelsTrain, miniBatchSize = 32):
	totalLoss = 0.0
	totalAccuracy = 0.0
	nmbSamples = len(imagesTrain)
	randomIndex = torch.randperm(nmbSamples)
	imagesTrain = imagesTrain[randomIndex] # Randomly shuffle data
	labelsTrain = labelsTrain[randomIndex] # dito
	for mbi in range(0, nmbSamples, miniBatchSize): # Update per minibatch
		optim.zero_grad() # Clear previous gradients
		batch  = imagesTrain[mbi:mbi+miniBatchSize] # minibatch
		labels = labelsTrain[mbi:mbi+miniBatchSize] # minibatch labels
		prediction = net(batch) # Make prediction of input
		loss = F.binary_cross_entropy(prediction, labels, reduction='sum')
		loss.backward() # Calculate gradients
		optim.step() # Update gradients
		totalLoss += float(loss) 
		totalAccuracy  += calcAccuracy(prediction, labels, False)
	# Return average loss and accuracy
	return totalLoss/nmbSamples, totalAccuracy/nmbSamples
						
net = Net() # initiate neural network
optim = torch.optim.Adam(net.parameters()) # the optimizer for the gradients

for epoch in range(int(1e1)):
	trainLoss, trainAccuracy = trainOneEpoch(imagesTrain, labelsTrain)
	testLoss, testAccuracy   = calculateTestLoss(imagesTest, labelsTest)
	print('\nEpoch: %i' % (epoch+1))
	print('Train loss: %.4f' % (trainLoss))
	print('Train accuracy %.2f%%' % (trainAccuracy*100))
	print('Test loss: %.4f' % testLoss)
	print('Test accuracy %.2f%%' % (testAccuracy*100))
	

