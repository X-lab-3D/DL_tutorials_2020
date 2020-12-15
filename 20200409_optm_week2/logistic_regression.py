# logistic regression instead of SVM -> also linear model but the group knows logistic regression and does not know SVM

# Basic imports
import torch
import torchvision
import torch.nn.functional as F
import numpy as np
from scipy import misc
from sklearn.metrics import confusion_matrix as confusion

torch.manual_seed(1234)

# Download data
dset = torchvision.datasets.MNIST('.', download=True)
images = dset.data
labels = dset.targets

# Pepare data
images = images.reshape(images.shape[0], -1) # Make the data flat, from 2dimensional to 1dimensional data
images = images/255. # Transforms data to float values between 0 and 1
oneHot = torch.zeros(images.shape[0], 10) # There are 10 possible outputs
for i,n in enumerate(labels):
    oneHot[i, n] = 1 # assign the correct output to the images
images = torch.Tensor(images).float() # transform numpy array to torch tensor

tp = .8
randomIndex = torch.randperm(images.shape[0])
imagesTrain = images[randomIndex[:int(tp*len(randomIndex))]]
imagesTest  = images[randomIndex[int(tp*len(randomIndex)):]]
oneHotTrain = oneHot[randomIndex[:int(tp*len(randomIndex))]]
oneHotTest  = oneHot[randomIndex[int(tp*len(randomIndex)):]]

# Initialize weights
W = torch.randn(images.shape[1], 10, requires_grad=True) # Weights
b = torch.randn(10, requires_grad=True) # bias

# Function that creates output
def logisticR(images, weights, bias):
    # Linear part
    output = torch.matmul(images, weights) + bias
    # Non-linear part
    output = F.softmax(output, 1)
    return output

# Mean squared error loss function, can be modified by group if they want
def lossFunction1(prediction, true_labels):
    return ((prediction-true_labels)**2).sum() + (W**2).sum()*0.01

# computing the gradients and apply them to the weights
def optimize(loss, weights, bias, learning_rate, print_gradients=False):
    try:
        weights.grad *= 0
        bias.grad *= 0
    except: pass
    # Calculate gradients
    loss.backward()
    # print gradients
    if print_gradients:
        print('Here are the gradients:')
        print(weights.grad)
    # Take step in negative gradient
    weights.data -= learning_rate*weights.grad
    bias.data -= learning_rate*bias.grad.data
    # Empty previous gradients
    weights.grad *= 0
    bias.grad *= 0

# Calculates how many % is correctly labeled in the dataset
def calcAccuracy(predictions, labels):
    return (predictions.argmax(1)==labels.argmax(1)).sum()/float(len(predictions)) * 100

# Different random initializations of the weights give different scores:
def random_initializations(times=10):
    startaccuracy = 0
    for time in range(times):
        W = torch.randn(images.shape[1], 10, requires_grad=True) # Weights
        b = torch.randn(10, requires_grad=True) # bias
        prediction = logisticR(images, W, b)
        accuracy = calcAccuracy(prediction, oneHot)
        if accuracy > startaccuracy:
            startaccuracy = accuracy
        print ('init %i: %.2f%% correct, best was %.2f%%' % (time, accuracy, startaccuracy))

def save_img(W, epoch):
    w = W.clone().detach().numpy()
    nmbrs = [w[:,i].reshape(28,28)for i in range(10)]
    misc.imsave('letter%i.png' % epoch, np.concatenate(nmbrs))

print('Different random initializations of the weights can already give different accuracies:')
random_initializations()
print()


print('Now start with one initialization of the weights, and use the gradients to optimize them:')

# Training the logistic regression
# No train-test split here, just to quickly show how it works
if __name__ == "__main__":
    epoch = 0
    while 1:
        randomIndex = torch.randperm(len(imagesTrain))
        imagesTrain = imagesTrain[randomIndex]
        oneHotTrain = oneHotTrain[randomIndex]
        predTrain = logisticR(imagesTrain, W, b)
        predTest  = logisticR(imagesTest, W, b)
        trainA = calcAccuracy(predTrain, oneHotTrain)
        testA  = calcAccuracy(predTest,  oneHotTest)
        print('Epoch %i. train acc: %.2f%% test acc %.2f%%:'% (epoch, trainA, testA))
        #print(confusion(predTrain.argmax(1), oneHotTrain.argmax(1)))
        for batchIndex in range(0, len(imagesTrain), 32):
            # Take 32 random samples from the dataset
            minibatchImages = imagesTrain[batchIndex:batchIndex+32]
            minibatchLabels = oneHotTrain[batchIndex:batchIndex+32]

            # # Make a prediction
            prediction = logisticR(minibatchImages, W, b)
            # Calculate the loss (how good the predictions are)
            loss = lossFunction1(prediction, minibatchLabels)
            # Optimize the gradients
            optimize(loss, W, b,learning_rate=0.001)
        if epoch % 10 == 0:
            save_img(W, epoch)
        if epoch % 25 == 0:
            torch.save(W, 'W')
        epoch += 1


