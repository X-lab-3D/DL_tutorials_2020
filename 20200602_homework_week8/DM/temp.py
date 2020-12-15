import torch
import imageio
import torch.nn as nn
import torch.nn.functional as F
from vae_example_conv import VAE


def image2dataset(imageLocation, outputShape):
    tensor = torch.Tensor(imageio.imread(imageLocation))/255
    chopped_tens = torch.empty(0)
    for x in range(35):
        for y in range(35):
            minitens = []
            minitens.append(torch.stack([j[32*y:32*(y+1)] for j in tensor[x*32:(x+1)*32]])) #one emoji
            minitens = torch.stack(minitens)
            chopped_tens = torch.cat((minitens, chopped_tens))
    shapeT = chopped_tens.permute(0, -1, 1, 2)
    if outputShape == 'convolutional':
        pass
    elif outputShape == 'linear':
        shapeT = shapeT.reshape(1225, 3072)
    return shapeT
 
imageLocation = './dataset.png'
outputShape = 'convolutional'
#imagesTrain = image2dataset(imageLocation, outputShape) # put here the emoji dataset pytorch tensor with shape: (nmbr_emojis, 3, 32, 32)

tensor = torch.Tensor(imageio.imread(imageLocation))/255 # 1120, 1120, 3
chopped_tens = torch.empty(0)
for x in range(35):
    for y in range(35):
        minitens = []
        minitens.append(torch.stack([j[32*y:32*(y+1)] for j in tensor[x*32:(x+1)*32]])) #one emoji
        minitens = torch.stack(minitens)
        chopped_tens = torch.cat((minitens, chopped_tens))
shapeT = chopped_tens.permute(0, -1, 1, 2)
if outputShape == 'convolutional':
    pass
elif outputShape == 'linear':
    shapeT = shapeT.reshape(1225, 3072)
    
#%%
#  imageio.imsave('./try.png', shapeT[1000].permute(1, 2, 0))

net = VAE()
checkpoint = torch.load('checkpoint.pth.tar')
net.load_state_dict(checkpoint['state_dict'])
net.eval()
sad_latents = net(shapeT[463:464])[1]
joke_latents = net(shapeT[800:801])[1]

latents_1 = sad_latents
latents_2 = joke_latents
n_steps = 12
#def latent_decoder(latents_1, latents_2, n_steps = 12):
diff_tensor = latents_2 - latents_1
reconstr = net.decoder(latents_1).squeeze(0)
imageio.imsave('outputs/latent_inter_0.png', reconstr.permute(1,2,0).detach())
for step in range(n_steps):
    tensor = latents_1 + ((diff_tensor/n_steps)*(step +1))
    reconstr = net.decoder(tensor).squeeze(0)
    imageio.imsave('outputs/latent_inter_%s.png' %str(step +1), reconstr.permute(1,2,0).detach())
reconstr = net.decoder(latents_2).squeeze(0)
imageio.imsave('outputs/latent_inter_%s.png' %str(n_steps +1), reconstr.permute(1,2,0).detach())
        
