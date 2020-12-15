import torch
import imageio
import torchvision
import torch.nn as nn
import torch.nn.functional as F


# Download data
dset = torchvision.datasets.MNIST('.', download=True)
imagesTrain = dset.data/255.

class VAE(nn.Module):
    def __init__(self, hidden = 8):
        super(VAE, self).__init__()
        self.hidden = hidden # number of hidden neurons
        self.encoder = nn.Sequential(
                        nn.Linear(784, 512), nn.BatchNorm1d(512), nn.ELU(),
                        nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ELU(),
                        nn.Linear(256, hidden*2)) # '*2' because you have just as many 'means' as log variances
        self.decoder = nn.Sequential(
                        nn.Linear(hidden, 256), nn.BatchNorm1d(256), nn.ELU(),
                        nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ELU(),
                        nn.Linear(512, 784), nn.Sigmoid())

    def encode(self, x):
        z = self.encoder(x)
         # Why logvar? because now the output can be any range, while standard deviation should always be positive
        mu, logvar = z[:,:self.hidden], z[:,self.hidden:] 
        return mu, logvar

    # Generate a sample given the means and logvars
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn(std.shape)
        return mu + eps*std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    # To generate a deterministic output
    def test(self, x):
        z, _ = self.encode(x.view(-1, 784))
        return self.decode(z)


# VAE specific loss function
def specialLoss(recon_x, x, mu, logvar):
    # Reconstruction error:
    bce = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    # regularizer, comparing the encoders distribution with a standard normal distribution
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kld


def trainOneEpoch(imagesTrain, miniBatchSize = 32):
    totalLoss = 0.0
    nmbSamples = len(imagesTrain)
    randomIndex = torch.randperm(nmbSamples)
    imagesTrain = imagesTrain[randomIndex] # Randomly shuffle data
    for mbi in range(0, nmbSamples, miniBatchSize): # Update per minibatch
        optim.zero_grad() # Clear previous gradients
        batch  = imagesTrain[mbi:mbi+miniBatchSize] # minibatch
        prediction, mu, logvar = net(batch) # Make prediction of input
        loss = specialLoss(prediction, batch, mu, logvar)
        loss.backward() # Calculate gradients
        optim.step() # Update gradients
        totalLoss += float(loss) 
    # Return average loss and accuracy
    return totalLoss/nmbSamples
				
net = VAE() # initiate neural network
optim = torch.optim.Adam(net.parameters()) # the optimizer for the gradients

if __name__ == '__main__':
    for epoch in range(1000):
        trainLoss = trainOneEpoch(imagesTrain) # train one epoch (=one run over the total dataset)
        print('\nEpoch: %i' % (epoch+1))
        print('Train loss: %.4f' % (trainLoss))
        torch.save(net.state_dict(), 'model') # save model






	
	
