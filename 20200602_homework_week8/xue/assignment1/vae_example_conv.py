import torch
import imageio
import torch.nn as nn
import torch.nn.functional as F


imagesTrain = '' # put here the emoji dataset pytorch tensor with shape: (nmbr_emojis, 3, 32, 32)

class UnFlatten(nn.Module):
    def __init__(self):
        super(UnFlatten, self).__init__()
    def forward(self, x):
        return x.view(x.shape[0], -1, 1, 1)

class VAE(nn.Module):
    def __init__(self, hidden = 16):
        super(VAE, self).__init__()
        self.hidden = hidden # number of hidden neurons
        self.encoder  = nn.Sequential(
                        nn.Conv2d(3, 32, 3), nn.BatchNorm2d(32), nn.ELU(), nn.MaxPool2d(2),
                        nn.Conv2d(32, 64, 3), nn.BatchNorm2d(64), nn.ELU(), nn.MaxPool2d(2),
                        nn.Conv2d(64, 128, 3), nn.BatchNorm2d(128), nn.ELU(),
                        nn.Conv2d(128, 256, 4), nn.BatchNorm2d(256), nn.ELU(), nn.Flatten(),
                        nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ELU(),
                        nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ELU(),
                        nn.Linear(64, hidden*2)
                        )
        self.decoder  = nn.Sequential(
                        nn.Linear(hidden, 64), nn.BatchNorm1d(64), nn.ELU(),
                        nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ELU(),
                        nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ELU(), UnFlatten(),
                        nn.ConvTranspose2d(256, 128, 3, stride=2), nn.BatchNorm2d(128), nn.ELU(),
                        nn.ConvTranspose2d(128, 64, 3, stride=2), nn.BatchNorm2d(64), nn.ELU(),
                        nn.ConvTranspose2d(64, 32, 3, stride=2), nn.BatchNorm2d(32), nn.ELU(),
                        nn.ConvTranspose2d(32, 3, 4, stride=2), nn.Sigmoid(),)

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
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    # To generate a deterministic output
    def test(self, x):
        self.eval()
        with torch.no_grad():
            z, _ = self.encode(x)
            reconstruction = self.decode(z)
        self.train()
        return reconstruction


# VAE specific loss function
def specialLoss(recon_x, x, mu, logvar):
    # Reconstruction error:
    bce = F.binary_cross_entropy(recon_x, x, reduction='sum')
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
    for epoch in range(10000):
        trainLoss = trainOneEpoch(imagesTrain) # train one epoch (=one run over the total dataset)
        print('\nEpoch: %i' % (epoch+1))
        print('Train loss: %.4f' % (trainLoss))
        if epoch % 10 ==0:
            torch.save({'state_dict': net.state_dict()}, 'checkpoint.pth.tar') # save model








