#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 15:51:56 2020

@author: farzanehmeimandi
"""

  
import torch
import imageio
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

#############################################################################
##Assignment 1
#############################################################################
## Problem 1.1
imageLocation = './dataset.png'

def image2dataset(imageLocation, outputShape):
    
    image_data = imageio.imread(imageLocation)
    print(image_data.shape)
    x = torch.Tensor(image_data)
    #Change dimension to 32*32
    a =x.unfold(0,32,32).unfold(1,32,32)
    
    
    if outputShape == 'convolutional':
        a = a.reshape(1225,3,32,32)
        a = a/255.
        # Remove all-white pictures
        T=a[0,:,:,:].unsqueeze(0)
        for i in range(1,1225):
            if a[i,:,:,:].sum(0).sum(0).sum(0) != torch.tensor(3*32*32):
                T = torch.cat([T , a[i,:,:,:].unsqueeze(0)])
        return T
    
    if outputShape == 'linear':
        a = a.reshape(1225,3072)
        a = a/255.
        # Remove all-white pictures
        T=a[0,:].unsqueeze(0)
        for i in range(1,1225):
            if a[i,:].sum(0) != torch.tensor(3072):
                T = torch.cat([T , a[i,:].unsqueeze(0)])
                
        return T

imagesTrain =  image2dataset(imageLocation, 'convolutional')# put here the emoji dataset pytorch tensor with shape: (nmbr_emojis, 3, 32, 32)
 

# VAE
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

#if __name__ == '__main__':
#    for epoch in range(10000):
#        trainLoss = trainOneEpoch(imagesTrain) # train one epoch (=one run over the total dataset)
#        print('\nEpoch: %i' % (epoch+1))
#        print('Train loss: %.4f' % (trainLoss))
#        if epoch % 10 ==0:
#            torch.save({'state_dict': net.state_dict()}, 'checkpoint.pth.tar') # save model
#

######################################
# Problem 1.2

net = VAE() # initiate neural network
checkpoint = torch.load('checkpoint.pth.tar')
net.load_state_dict(checkpoint['state_dict'])

# Checked graphically

#######################################
# Problem 1.3

def interpolate_two_pics():
    net = VAE() 
    checkpoint = torch.load('./checkpoint.pth.tar')
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()
    
    index_1 = 385
    index_2 = 981
    image_1=  imagesTrain[index_1].permute([1,2,0])
    image_2=  imagesTrain[index_2].permute([1,2,0])
    
    # Original pixel space
    imageio.imsave(str(index_1)+".png", image_1)
    imageio.imsave(str(index_2)+".png", image_2)


    interpolated_ori = (image_1 + image_2)/2
    imageio.imsave(str(index_1)+"_"+str(index_2)+".png", interpolated_ori)

    # Latent space
    mu_1, logvar_1 = net.encode(image_1.permute(2,0,1).unsqueeze(0))
    mu_2, logvar_3 = net.encode(image_2.permute(2,0,1).unsqueeze(0))
    new_mu = (mu_2 + mu_1)/2

    new_Image = net.decode(new_mu).permute(0, 2, 3, 1).detach()
    imageio.imsave(str(index_1)+"_"+str(index_2)+"_latent.png", new_Image.squeeze(0))


interpolate_two_pics()

##########################################
# Problem 1.4

def interpolate_three_pics():
    net = VAE() 
    checkpoint = torch.load('./checkpoint.pth.tar')
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()
    
    mu_a = interpolate_one_pic('a.png',net)
    mu_b = interpolate_one_pic('b.png',net)
    mu_c = interpolate_one_pic('c.png',net)
    mu_d = interpolate_one_pic('d.png',net)
    mu_e = interpolate_one_pic('e.png',net)
    
    mu_b = mu
    mu_eq1 = mu_a - mu_b + mu_c
    new_pic = net.decode(mu_eq1).permute(0, 2, 3, 1).detach() 
    imageio.imsave("eq1_latent.png", new_pic.squeeze(0))
    
    mu_eq2 = mu_a - mu_b + mu_d
    new_pic = net.decode(mu_eq2).permute(0, 2, 3, 1).detach()
    imageio.imsave("eq2_latent.png", new_pic.squeeze(0))
    
    mu_eq3 = mu_a - mu_b + mu_e
    new_pic = net.decode(mu_eq3).permute(0, 2, 3, 1).detach()
    imageio.imsave("eq3_latent.png", new_pic.squeeze(0))
    


def interpolate_one_pic(file_name,net):
    pic = imageio.imread(file_name)
    tensor_pic = torch.Tensor(pic)[:,:,:3].permute(2, 0, 1)/255.
    mu, logvar = net.encode(tensor_pic.unsqueeze(0))

    return mu

interpolate_three_pics()


##########################################
# Problem 1.5

# Step 1
def newLoss(recon_x, x):
    # Reconstruction error:
    bce = F.binary_cross_entropy(recon_x, x, reduction='sum')
    return bce


net = VAE() 
checkpoint = torch.load('./checkpoint.pth.tar')
net.load_state_dict(checkpoint['state_dict'])
net.eval()

# method 1
reconstructed_images = net.test(imagesTrain)
loss= torch.Tensor(len(reconstructed_images))

for i in range(len(reconstructed_images)):
    loss[i]=newLoss(reconstructed_images[i,:,:,:].squeeze(0), imagesTrain[i,:,:,:].squeeze(0))
 
sorted_loss, indices_loss = torch.sort(loss)
best  = imagesTrain[indices_loss[:10]].permute([0,2,3,1])
best_recon  = reconstructed_images[indices_loss[:10]].permute([0,2,3,1])


worst = imagesTrain[indices_loss[-10:]].permute([0,2,3,1])
worst_recon  = reconstructed_images[indices_loss[-10:]].permute([0,2,3,1])

best_concatanated  =  best.reshape(10*32, 32, 3)
best_recon_concatanated = best_recon.reshape(10*32, 32, 3)
best_all = torch.cat([best_concatanated,best_recon_concatanated],dim=1)

worst_concatanated = worst.reshape(10*32, 32, 3)
worst_recon_concatanated = worst_recon.reshape(10*32, 32, 3)
worst_all = torch.cat([worst_concatanated,worst_recon_concatanated],dim=1)

imageio.imsave("best_method1.png", best_all )
imageio.imsave("worst_method1.png", worst_all)


# method 2
avg_image = torch.mean(imagesTrain,dim=0)

Eucl_dist = torch.Tensor(len(imagesTrain))
for i in range(len(imagesTrain)):
    Eucl_dist[i] = torch.dist(reconstructed_images[i,:,:,:].squeeze(0), avg_image, 2)


sorted_loss, indices_loss = torch.sort(Eucl_dist)
best  = imagesTrain[indices_loss[:10]].permute([0,2,3,1])
best_recon  = reconstructed_images[indices_loss[:10]].permute([0,2,3,1])


worst = imagesTrain[indices_loss[-10:]].permute([0,2,3,1])
worst_recon  = reconstructed_images[indices_loss[-10:]].permute([0,2,3,1])

best_concatanated  =  best.reshape(10*32, 32, 3)
best_recon_concatanated = best_recon.reshape(10*32, 32, 3)
best_all = torch.cat([best_concatanated,best_recon_concatanated],dim=1)

worst_concatanated = worst.reshape(10*32, 32, 3)
worst_recon_concatanated = worst_recon.reshape(10*32, 32, 3)
worst_all = torch.cat([worst_concatanated,worst_recon_concatanated],dim=1)

imageio.imsave("best_method2.png", best_all )
imageio.imsave("worst_method2.png", worst_all)


###########################################################################
## Assignment 2
###########################################################################


model = nn.Sequential (
        nn.Conv2d(in_channels = 3, out_channels, kernel_size=1),
        activation_function,
        nn.Conv2d(in_channels, out_channels=1, kernel_size =1)
        nn.Sigmoid()
        )