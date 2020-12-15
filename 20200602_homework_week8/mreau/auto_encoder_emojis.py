import torch
import imageio
import torch.nn as nn
import torch.nn.functional as F

### Assignment 1.1
def image2dataset(imageLocation, outputShape):
        im = imageio.imread(imageLocation)
        im = torch.Tensor(im)
        if outputShape == 'convolutional':
            all_im = []
            for i in range(0, im.size()[0], 32):
                for j in range(0, im.size()[1], 32):
                    im_ij = im[i:i+32,j:j+32,:]
                    im_ij = im_ij.unsqueeze(0).permute(0,3,1,2)
                    if torch.sum(im_ij) != 32*32*3*255 : # check for white images
                        all_im.append(im_ij)
            dataset = torch.cat(all_im)
            dataset = dataset/ 255.0
            return dataset
        if outputShape == 'linear':
            dataset = imt.view(-1, 3072)
            dataset = dataset/ 255.0
            return dataset

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

# LOAD DATA
dataset = image2dataset('dataset.png', 'convolutional')
imagesTrain = dataset # put here the emoji dataset pytorch tensor with shape: (nmbr_emojis, 3, 32, 32)
   
net = VAE() # initiate neural network
checkpoint = torch.load('checkpoint.pth.tar')
net.load_state_dict(checkpoint['state_dict'])
optim = torch.optim.Adam(net.parameters()) # the optimizer for the gradients

### Assignment 1.3
a = torch.Tensor(imageio.imread('a.png'))
b = torch.Tensor(imageio.imread('b.png'))
c = torch.Tensor(imageio.imread('c.png'))
d = torch.Tensor(imageio.imread('d.png'))
e = torch.Tensor(imageio.imread('e.png'))

im_a = a.permute(2,0,1).unsqueeze(0)/255.
im_b = b.permute(2,0,1)[0:3].unsqueeze(0)/255.
im_c = c.permute(2,0,1).unsqueeze(0)/255.
im_d = d.permute(2,0,1).unsqueeze(0)/255.
im_e = e.permute(2,0,1).unsqueeze(0)/255.

encode = net.encode
decode = net.decode
test = net.test

def test_interpolate(x, y, prop_x = 0.5):
        net.eval()
        with torch.no_grad():
            # interpolate emojis in the pixel space
            xy = (x*prop_x + y*(1-prop_x))
            zxy, _ = encode(xy)
            pixel = decode(zxy)
            # interpolate emojis in the latent space
            zx, _ = encode(x)
            zy, _ = encode(y)
            z = (zx*prop_x + zy*(1-prop_x))
            latent = decode(z)
        return latent, pixel

def saveimage(im, name="reconstruction"):
    im = im.permute(0,2,3,1)
    im_reshaped = im.flatten(start_dim=0, end_dim=1)
    im_col = im_reshaped.numpy()*255
    imageio.imsave("output/{}.png".format(name), im_col)

for i in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
    latent, pixel = test_interpolate(im_c,im_d,i)
    saveimage(latent, "latent_cd_{}".format(i))
    saveimage(pixel, "pixel_cd_{}".format(i))

### Assignment 1.4
net.eval()

def combine(a,b,c, name='combine'):
    with torch.no_grad():
        za, _ = encode(a)
        zb, _ = encode(b)
        zc, _ = encode(c)
        latent_abc = decode(za-zb+zc)
        saveimage(latent_abc, "latent_{}".format(name))

combine(im_a,im_b,im_c, 'abc')
combine(im_a,im_b,im_d, 'abd')
combine(im_a,im_b,im_e, 'abe')

### Assignment 1.5
#Step 1
# check the difference between encoded images and dataset images
net.eval()

dataset = image2dataset('dataset.png', 'convolutional')
imagesTrain = dataset # put here the emoji dataset pytorch tensor with shape: (nmbr_emojis, 3, 32, 32)
recontructed_dataset = test(imagesTrain)
imageio.imread('dataset.png').shape

quality_1 = []
# compute Euclidean distance
for i, im in enumerate(recontructed_dataset) :
    e1 = (dataset[i] - recontructed_dataset[i])**2
    e2 = torch.sum(e1)
    euclidian = torch.sqrt(e2)
    quality_1.append(euclidian.item())

sorted_index_1 = sorted(range(len(quality_1)), key=lambda k: quality_1[k])
best_quality_1 = dataset[sorted_index_1[0:10]]
worst_quality_1 = dataset[sorted_index_1[-10:]]

saveimage(best_quality_1, 'best_quality')
saveimage(worst_quality_1, 'worst_quality')

#Step 2
# check the difference between the average image and original dataset images
net.eval()

avg_emoji = torch.mean(dataset,dim=0)

quality_2 = []
# compute Euclidean distance
for i, im in enumerate(recontructed_dataset) :
    e1 = (dataset[i] - avg_emoji)**2
    e2 = torch.sum(e1)
    euclidian = torch.sqrt(e2)
    quality_2.append(euclidian.item()) 

sorted_index_2 = sorted(range(len(quality_2)), key=lambda k: quality_2[k])
best_quality_2 = dataset[sorted_index_2[0:10]]
worst_quality_2 = dataset[sorted_index_2[-10:]]

saveimage(best_quality_2, 'best_quality_avg')
saveimage(worst_quality_2, 'worst_quality_avg')
