import torch
import imageio
from sripts_and_files_for_homework.vae_example_conv import VAE
from torch.nn import functional as F
import matplotlib.pyplot as plt
from random import randint

net = VAE(16)
net.load_state_dict(torch.load("sripts_and_files_for_homework/checkpoint.pth.tar")['state_dict'])
net.eval()

"""
+++ gen images +++
"""

def image2dataset(location, outshape = ""):
    img = torch.tensor(imageio.imread(location, 'png'))
    row_emojis = img.shape[0]/32
    col_emojis = img.shape[1]/32

    NR_emojis =int(row_emojis * col_emojis)

    emojis = torch.Tensor()

    r_counter = 0
    c_counter = 0
    for emojiNR in range(NR_emojis):
        emoji = img[r_counter*32 : (r_counter+1)*32 , c_counter*32 : (c_counter+1)*32]
        plt.imshow(emoji)
        plt.show()
        emojis = torch.cat((emojis,emoji.float().unsqueeze(0)), dim=0)
        if c_counter == col_emojis-1:
            c_counter=0
            r_counter+=1
        else:
            c_counter+=1

    return emojis/255 #between 0 and 1
try: images = torch.load("imawwwges")
except:
    images = image2dataset("sripts_and_files_for_homework/dataset.png")
    torch.save(images, "images")

"""
+++ Inerpolate +++
"""

def interpolate(imgA, imgB, space = 'latent', steps = 11):

    imgs = torch.Tensor()
    if space == 'latent':
        muA, logvarA = net.encode(imgA.permute(2,0,1).unsqueeze(0))
        muB, logvarB = net.encode(imgB.permute(2,0,1).unsqueeze(0))
        diff_mu = muB - muA
        stepMu = diff_mu / steps


        for step in range(steps + 1):
            mu = muA + stepMu * step
            newImg = net.decode(mu).permute(0, 2, 3, 1).detach()
            imgs = torch.cat((imgs, newImg))

    elif space == 'pixel':
        diff_imgs = imgB - imgA
        stepMu = diff_imgs / steps

        imgs = torch.Tensor()
        for step in range(steps + 1):
            newImg = imgA + stepMu * step
            imgs = torch.cat((imgs, newImg))
    return imgs

if 0:
    for i in range(10):
        A =randint(0,images.shape[0])
        B =randint(0,images.shape[0])
        imgA = images[A]
        imgB = images[B]
        extrapolates_latent = interpolate(imgA, imgB)
        extrapolates_pixel = interpolate(imgA, imgB, 'pixel')
        plt.imshow(extrapolates_latent.view(-1,32,3))
        plt.savefig("imagesFolder/%s-%s_lat"%(A,B))
        plt.imshow(extrapolates_pixel.view(-1,32,3))
        plt.savefig("imagesFolder/%s-%s_pix"%(A,B))

"""
+++ Substraction +++
"""

if 0:
    # substract emojis
    A, _ = net.encode(torch.tensor(imageio.imread("sripts_and_files_for_homework/a.png", 'png')/255).float().unsqueeze(0).permute(0,3,1,2))
    B, _ = net.encode(torch.tensor(imageio.imread("sripts_and_files_for_homework/b.png", 'png')/255).float()[:,:,:3].unsqueeze(0).permute(0,3,1,2))
    C, _ = net.encode(torch.tensor(imageio.imread("sripts_and_files_for_homework/c.png", 'png')/255).float().unsqueeze(0).permute(0,3,1,2))
    D, _ = net.encode(torch.tensor(imageio.imread("sripts_and_files_for_homework/d.png", 'png')/255).float().unsqueeze(0).permute(0,3,1,2))
    E, _ = net.encode(torch.tensor(imageio.imread("sripts_and_files_for_homework/e.png", 'png')/255).float().unsqueeze(0).permute(0,3,1,2))

    ans1 = A - B + C
    ans2 = A - B + D
    ans3 = A - B + E
    imageio.imwrite("/mnt/sda3/My_data/Jarek/Documents/Studie/2019-2020/stage/deep_learning_homework/week 8/sripts_and_files_for_homework/ans1.png", net.decode(ans1)[0].permute(2, 1, 0).detach())
    imageio.imwrite("/mnt/sda3/My_data/Jarek/Documents/Studie/2019-2020/stage/deep_learning_homework/week 8/sripts_and_files_for_homework/ans2.png", net.decode(ans2)[0].permute(2, 1, 0).detach())
    imageio.imwrite("/mnt/sda3/My_data/Jarek/Documents/Studie/2019-2020/stage/deep_learning_homework/week 8/sripts_and_files_for_homework/ans3.png", net.decode(ans3)[0].permute(2, 1, 0).detach())

"""
+++ Anomality +++
"""

# step 1
if 0:
    reconstructions = net.test(images.permute(0,3,1,2))
    bce = F.binary_cross_entropy(reconstructions, images.permute(0,3,1,2), reduction='none')
    bceSum = bce.sum(dim=[1,2,3])
    highests, higidx = torch.topk(bceSum, 10)
    lowest, lowidx = torch.topk(bceSum, 10, largest=False)

    highestIms, higestRecon = images.permute(0,3,1,2)[higidx], reconstructions[higidx]
    highestIms = highestIms.permute(0,2,3,1).view(-1,32,3)
    higestRecon = higestRecon.permute(0,2,3,1).reshape(-1,32,3)

    lowestIms, lowestRecon = images.permute(0,3,1,2)[lowidx], reconstructions[lowidx]
    lowestIms = lowestIms.permute(0,2,3,1).view(-1,32,3)
    lowestRecon = lowestRecon.permute(0,2,3,1).reshape(-1,32,3)

    imrecon = torch.cat((highestIms,higestRecon, lowestIms, lowestRecon), dim=1)

    plt.imshow(imrecon)
    plt.title('best and worst reconstructions')
    plt.savefig('bestAndWorst')

#step 2
meanEmoji = torch.mean(images, dim=0)
print(meanEmoji.shape)
# plt.imshow(meanEmoji)
# plt.show()
avgPixDist = ((images-meanEmoji)**2).sqrt().sum(dim=[1,2,3])

highests, higidx = torch.topk(avgPixDist, 10)
lowest, lowidx = torch.topk(avgPixDist, 10, largest=False)

highestIms, lowestIms = images.permute(0,3,1,2)[higidx], images.permute(0,3,1,2)[lowidx]

highestIms = highestIms.permute(0, 2, 3, 1).view(-1, 32, 3)
lowestIms = lowestIms.permute(0, 2, 3, 1).view(-1, 32, 3)

imrecon = torch.cat((lowestIms, highestIms), dim=1)
plt.imshow(imrecon)
plt.title('least and most from average')
plt.savefig('bestAndWorstAveraged')
