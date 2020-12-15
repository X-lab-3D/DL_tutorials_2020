import torch
from skimage import io, transform
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

BLUE = torch.tensor([0,0,1])
RED = torch.tensor([1,0,0])
IMSIZE = 150
nn = torch.nn
F = torch.nn.functional

"""##########
Loading images
############"""
def loadImages(path):
    """
    :param path:
    :return: image in form: (N,IMSIZE, IMSIZE, 3). Values are RGB and between 0-1
    """
    imageNames = os.listdir(path)
    images = torch.zeros(len(imageNames), IMSIZE,IMSIZE,3)
    for i, im in enumerate(imageNames):
        im = io.imread(path+im)
        im = transform.resize(im,(IMSIZE,IMSIZE))
        try:
            images[i] = torch.tensor(im)[:,:,:3] #in case of png and/or transparent layer
        except Exception as e:
            print(e)
            print(path + imageNames[i])

    return images


"""##########
defining network and functions
############"""
net = nn.Sequential(    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1),
                        nn.ELU(),
                        nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1),
                        nn.Sigmoid()
)

trainer = torch.optim.Adam(net.parameters())

def train(images, labels):
    net.train()
    trainer.zero_grad()
    images = images.permute(0,3,1,2)
    out = net(images).mean(dim=[1,2,3])
    loss = F.binary_cross_entropy(out, labels)
    loss.backward()
    trainer.step()

def test(images):
    net.eval()
    finImage = torch.Tensor()
    for i, im in enumerate(images):
        im2 = im.permute(2,0,1).unsqueeze(0)
        out = net(im2).detach()[0].permute(1,2,0).view(IMSIZE*IMSIZE)
        # out = torch.ones_like(out) * 0.2

        higlighted = turnColor(out, BLUE, RED)
        # higlighted = torch.zeros(IMSIZE,IMSIZE,3)
        # for kk in range(3):
        #     higlighted[:,:,kk] = out.view(IMSIZE,IMSIZE)

        thisIm = torch.cat((im,higlighted), dim=1)
        finImage = torch.cat((finImage, thisIm), dim = 0)
    finImage = finImage * 255
    finImage = finImage.int().detach().numpy().astype("uint8")

    return finImage

def turnColor(odds, c1, c2):
    """

    :param odds:
    :param c1: color for non-skin
    :param c2: color for skin
    :return:
    """

    #TODO: normalize values ?
    middle = 0.5

    image = torch.zeros(IMSIZE * IMSIZE, 3)

    white = torch.tensor([1, 1, 1])

    c1idx = odds < middle
    c2idx = odds >= middle

    noSkinOdds = 1 - (odds[c1idx].view(-1,1) * 2) # very low odds -> very likely to be no skin
    skinOdds = (odds[c2idx].view(-1,1) - middle) * 2 # very high odds -> very likely to be skin

    c1pixels = noSkinOdds * c1 + (1 - noSkinOdds) * white
    c2pixels = skinOdds * c2 + (1 - skinOdds) * white

    image[c1idx] = c1pixels
    image[c2idx] = c2pixels

    return image.view(IMSIZE, IMSIZE, 3)


"""##########
running
############"""

eval = loadImages("data/skinVal/")
if input("reload images? ([y]es / any key for no): ") == 'y':
    noSkin = loadImages("data/noSkin/")
    skin = loadImages("data/skin/")
    torch.save(noSkin, 'noskinIms')
    torch.save(skin, 'skinIms')
else:
    noSkin = torch.load('noskinIms')
    skin = torch.load('skinIms')

all_data = torch.cat((skin,noSkin))
allLabs = torch.cat((torch.ones(skin.shape[0]), torch.zeros(noSkin.shape[0])))
random_idx = torch.randperm(skin.shape[0])[0:5]
randTrainsSkin = skin[random_idx]
random_idx = torch.randperm(noSkin.shape[0])[0:5]
randTrainsNoskin = noSkin[random_idx]

# for i in tqdm(range(1000)):
#     train(all_data, allLabs)
#     if i%10 == 0:
#         evalIm = test(eval)
#         io.imsave('skinIm/evals_{}.png'.format(i), evalIm)
#         io.imsave('skinIm/lastest_eval.png', evalIm)
#         trainIm = test(randTrainsSkin)
#         io.imsave('skinIm/skinTrain_{}.png'.format(i), trainIm)
#         io.imsave('skinIm/latest_train.png', trainIm)
#         trainIm2 = test(randTrainsNoskin)
#         io.imsave('skinIm/noSkinTrain_{}.png'.format(i), trainIm2)
#         io.imsave('skinIm/latest_nsktrain.png', trainIm2)
#         torch.save(net,"skinIm/net{}".format(i))

net = torch.load("skinIm/net990")
ims = loadImages("data/skinTest/")
result = test(ims)
io.imsave("TEST.png", result)
