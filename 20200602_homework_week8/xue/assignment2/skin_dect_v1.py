#!/usr/bin/env python
# Li Xue
#  3-Jul-2020 16:01

import os
import sys
import re
from os import listdir
import shutil
import glob
from PIL import Image
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torch.utils.data as data_utils
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import subprocess
import pdb

def resize_imgs(img_dir):

    # create img_dir/ori
    img_dir_ori = f"{img_dir}/ori"
    if not os.path.isdir(img_dir_ori):
        os.mkdir(img_dir_ori)
    # resize img and move the original to img_dir/ori
    imgFLs = [f for f in glob.glob(f"{img_dir}/*") if re.search('(jpg|png|jpeg)',f, flags=re.IGNORECASE) ]

    i = 0
    for imgFL in imgFLs:
        i = i+1
        img = Image.open(imgFL)
        img_new = img.resize((150,150))
        try:
            shutil.move(imgFL, f"{img_dir_ori}")
        except:
            # the original imgFL is already save in img_dir_ori
            os.remove(imgFL)

        newFL = os.path.splitext(imgFL)[0] + '.jpg'
        try:
            img_new.save(newFL)
        except:
            print(f"Error: cannot save {imgFL}")

    print(f"Resized imgs under: {img_dir}/xx.jpg")

resize_imgs('Dataset/train/nonSkinPhoto')
resize_imgs('Dataset/train/SkinPhoto')
resize_imgs('Dataset/validation/neg')
resize_imgs('Dataset/validation/pos')
resize_imgs('Dataset/test/neg')
resize_imgs('Dataset/test/pos')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.bn0 = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(3, 32, (1,1), stride = 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,1, (1,1), stride = 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        x = self.conv1(x)
        x = F.elu(x)
        #x = self.bn1(x) #used to make relu stable
        #x = F.relu(x) # relu without bn1 does not work well: tends to generate too many black (0,0,0) pixels
        x = self.conv2(x)
        y = self.sigmoid(x)
        return y
net = Net()
summary(net, input_size=(3, 150, 150))

def get_Flnames(img_dir_pos, img_dir_neg):
    # return
    #   1. a list of image files (path + names)
    #   2. labels

    imgFLs_pos = [f"{img_dir_pos}/{f}" for f in os.listdir(img_dir_pos) if re.search('.+.jpg',f)]
    imgFLs_neg = [f"{img_dir_neg}/{f}" for f in os.listdir(img_dir_neg) if re.search('.+.jpg',f)]
    imgFLs = imgFLs_pos +  imgFLs_neg
    num_pos = len(imgFLs_pos)
    num_neg = len(imgFLs_neg)
    labels = [1] * num_pos + [0] * num_neg
    print(f"\n{num_pos} positive imgs read from {img_dir_pos}")
    print(f"{num_neg} negtive imgs read from {img_dir_neg}\n")
    return imgFLs, labels

class myDataset(data_utils.Dataset):
    def __init__(self, imgFLs, labels):
        self.imgFLs = imgFLs
        self.labels = labels
    def __len__(self):
        return len(self.imgFLs)
    def __getitem__(self, idx):
        img_name = self.imgFLs[idx]
        label = self.labels[idx]
        #print(f"Reading --> idx: {idx}, img_name: {img_name}, label: {label}")
        image = imageio.imread(img_name)
        image = torch.Tensor(image).permute([2,0,1])
        image = image/255
        return image, label


def train_one_epo(net, data_loader):
    net.train()
    losses = []
    for x, targets in data_loader:
        optimizer.zero_grad()
        y = net(x)
        criterion = nn.BCELoss()
        loss = criterion(torch.mean(y,dim=[1,2,3]), targets.to(torch.float))
        loss.backward()
        losses.append(loss)
        optimizer.step()

    loss_ave = sum(losses)/len(losses)
    return net, loss_ave

def evaluate(net, data_loader):

    net.eval()
    losses = []
    with torch.no_grad():
        for x, targets in data_loader:
            y = net(x)
            criterion = nn.BCELoss()
            loss = criterion(torch.mean(y,dim=[1,2,3]), targets.to(torch.float))
            losses.append(loss)

        loss_ave = sum(losses)/len(losses)
    return loss_ave

def prepare_dataset(img_dir_pos, img_dir_neg, batch_size):
    imgFLs, labels = get_Flnames(img_dir_pos, img_dir_neg)

    train_dataset = myDataset(imgFLs , labels )
    index = list(range(train_dataset.__len__()))
    data_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=2)
    return data_loader

def train(net, epoches, data_loader_train, data_loader_eval):
    losses_train = torch.Tensor(epoches)
    losses_eval = torch.Tensor(epoches)

    for epoch in range(epoches):
        mini_batch = 0
        net, loss_train = train_one_epo(net, data_loader_train)

        loss_eval = evaluate(net, data_loader_eval)
        print(f"epoch = {epoch}, loss_train = {loss_train:.4f}, loss_eval = {loss_eval:.4f}")
        losses_train[epoch] = loss_train
        losses_eval[epoch] = loss_eval

        #-- save model
        if not os.path.isdir('networks'): os.mkdir('networks')
        modelFL = f"networks/model_epo{epoch}.pt"
        torch.save(net.state_dict(), modelFL)
        print(f"--> One epoch finished. Modeled saved: {modelFL}")


        if epoch % 1 == 0:
            if not os.path.isdir("pred_imgs"): os.mkdir("pred_imgs")
            outputFL = f"pred_imgs/eval_{epoch}.png"
            visual_check(modelFL, 'Dataset/validation/pos/' , outputFL)
            outputFL = f"pred_imgs/train_{epoch}.png"
            visual_check(modelFL, 'Dataset/train/SkinPhoto/' , outputFL)

    return net, losses_train, losses_eval

def plot_loss(losses_train, losses_eval):

    # save as torch file
    torch.save(losses_train, "losses_train.pt")
    torch.save(losses_eval, "losses_eval.pt")

    # save as tsv file
    losses = torch.stack( (losses_train, losses_eval), dim = 1).detach().numpy()
    losses = pd.DataFrame(losses, columns = ['train', 'eval'])
    losses.to_csv('losses.tsv', sep = '\t', index = False)
    print(f"losses.tsv generated.")

    # generate plots
    subprocess.check_call(['Rscript', 'plot_losses.R', 'losses.tsv'])

def black_white(im_new, cutoff = 0.5):
    #- convert image (original range 0-1) to black and white
    idx1 = im_new < cutoff
    idx2 = im_new >= cutoff
    im_new[idx1] = 0
    im_new[idx2] = 1
    return im_new

def visual_check(networkFL, img_dir, outputFL):
    # save predicted images into png files
    net = Net()
    net.load_state_dict(torch.load(networkFL))

    # visually check the valiation set
    net.eval()
    imgFLs = [f"{img_dir}/{f}" for f in os.listdir(img_dir) if os.path.isfile(f"{img_dir}/{f}")]
    n_imgs = len(imgFLs)
    print(f"There are {n_imgs} images under {img_dir}")

    with torch.no_grad():
        images = torch.Tensor().to(torch.uint8)
        for i in range(len(imgFLs)):
            im = imageio.imread(imgFLs[i])
            im_new = net(torch.Tensor(im).unsqueeze(0).permute(0,3,1,2))
            im_new = (im_new*255).squeeze().to(torch.uint8)

            #--
            im_new = torch.stack((im_new, im_new, im_new), dim = 2) #im_new.shape = [150, 150] -> im_new.shape = [150, 150, 3]
            im = torch.Tensor(im).to(torch.uint8)
            this_im = torch.cat((im, im_new), dim = 1) # put the original image and pred image side-by-side
            images = torch.cat((images, this_im), dim = 0)

    # save into file
    imageio.imsave(outputFL,images.to(torch.uint8).numpy())
    print(f"{outputFL} generated.")

#-- train and evaluate
data_loader_train = prepare_dataset(img_dir_pos = 'Dataset/train/SkinPhoto',
        img_dir_neg = 'Dataset/train/nonSkinPhoto', batch_size = 5)
data_loader_eval = prepare_dataset(img_dir_pos = 'Dataset/validation/pos',
        img_dir_neg = 'Dataset/validation/neg', batch_size = 20)
net = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")
net.to(device)
#net.load_state_dict(torch.load('networks/old/model_epo299.pt'))
optimizer = torch.optim.Adam(net.parameters())

epoches = 50
net, losses_train, losses_eval = train(net,epoches, data_loader_train, data_loader_eval)
plot_loss(losses_train, losses_eval)

#- the model with lowest eval loss (the loss does not make much sense here ...)
loss_min , idx = torch.min(losses_eval, 0)
networkFL = f'neworks/model_epo{idx}.pt'
print(f"Model with the lowest eval loss: {networkFL}, epoch = {idx}, and loss_eval = {loss_min:4f}")

sys.exit()
# networkFL = 'networks/model_epo11.pt'
# outputFL = f"pred_imgs/test_{11}.png"
# visual_check(networkFL, 'Dataset/test/pos/' , outputFL)



