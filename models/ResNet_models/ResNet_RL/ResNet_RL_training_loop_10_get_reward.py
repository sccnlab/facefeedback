import pandas as pd
import pickle
from PIL import Image
import numpy as np
import os
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import shelve
import math
import numpy as np
import sys
import matplotlib.pyplot as plt
import scipy.stats as stats
import random
# import pickle5 as pickle

##dataloader
from torch.utils.data import Dataset, DataLoader

class FaceDataset(Dataset):
    def __init__(self, txt_path, img_dir, transforms = None):
        df = pickle.load(open(txt_path, "rb"))
        df['img'] = df['img'].str.replace("/mmfs1/data/schwarex/neuralNetworks/identity/datasets/img_align_celeba",
                                          img_dir)
        self.txt_path = txt_path
        self.y = df['new_label']
        self.x = df['img']
        self.n_samples = len(df)
        self.transforms = transforms

        self.mu = torch.randn(1503)
        m = torch.distributions.gamma.Gamma(torch.ones(1503), torch.ones(1503) * 20)
        self.sigma = m.sample()

    def __len__(self):
        return self.n_samples


    def __getitem__(self, index):
        image = Image.open(self.x[index]).convert('RGB')

        if self.transforms:
            image = self.transforms(image)

        label = self.y[index]
        base_reward = torch.randn(1)
        reward = base_reward * self.sigma[label] + self.mu[label]
        data = [image, label, reward]

        return data

## RGB images    
transform = transforms.Compose([transforms.CenterCrop(178),  #original image size = (178, 218)
                                transforms.Resize(128),
                                transforms.ToTensor()])       


train_dataset = FaceDataset(txt_path = '/data/zhouabx/celebA/train_SUBSET.pkl',
                            img_dir = '/mmfs1/data/zhouabx/celebA/img_align_celeba/',
                            transforms = transform)

trainloader = DataLoader(dataset = train_dataset, batch_size = 64, shuffle = True)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNetEncoder(nn.Module):
    def __init__(self, num_blocks, k):
        super(ResNetEncoder, self).__init__()
        self.in_channels = k
        self.conv1 = nn.Conv2d(3, k, kernel_size = 3, stride = 1, padding = 1, bias = False)  ## RGB images
        self.bn1 = nn.BatchNorm2d(k)
        self.layer1 = self._make_layer(k, num_blocks[0], stride = 1)       ## 64
        self.layer2 = self._make_layer(k * 2, num_blocks[1], stride = 2)   ## 64 * 2 = 128
        self.layer3 = self._make_layer(k * 4, num_blocks[2], stride = 2)   ## 64 * 4 = 256
        self.layer4 = self._make_layer(k * 8, num_blocks[3], stride = 2)   ## 64 * 8 = 512

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        
        third_last_features = out ## feature from third last layer
        
        out = self.layer3(out)
        
        second_last_features = out ## feature from second last layer
        
        out = self.layer4(out) ## feature from last layer
        return out, second_last_features, third_last_features
   
    
class ResNet18(nn.Module):
    def __init__(self, k):
        super(ResNet18, self).__init__()
        self.encoder = ResNetEncoder([2, 2, 2, 2], k) # four layers, each layer contains 2 residual blocks
                 
        ## q network
        self.qnetwork = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )          
    
    def q(self, z):
        return self.qnetwork(z)   
  
    def forward(self, x):
        feature_map, second_last_features, third_last_features = self.encoder(x)
        x = F.adaptive_avg_pool2d(feature_map, (1, 1))   ## global average pooling 
        x = x.view(x.size(0), -1)       
        qvalues = self.q(x)
        return qvalues, feature_map, second_last_features, third_last_features
 
ResModel = ResNet18(64) # standard ResNet-18
ResModel.cuda()

def customloss1D(qvalues, reward, epoch, min_q):
  batch_size = len(qvalues)
  loss = torch.zeros(batch_size).cuda()
  ps = F.softmax(torch.cat((torch.zeros(batch_size, 1).cuda(),qvalues),dim = 1),dim = 1).cuda()
  x = torch.rand(batch_size,1).cuda()
  for i in range(batch_size):
    if ps[i,1] > x[i,0]: # if the agent interacts with the object/face, they get a loss equal to the error at predicting the reward
      expectation = qvalues[i,0]
      prediction_error = (reward[i] - expectation)**2
      loss[i] = prediction_error
    else:              # if the agent does not interact with the object/face, they get a loss equal to the opportunity cost they think is associated with not interacting
      opportunity_cost = qvalues[i,0] + (1 - epoch/20) * (1 - min_q)
  return torch.sum(loss)
        
def get_reward(qvalues, reward):
  batch_size = len(qvalues)
  getreward = torch.zeros(batch_size).cuda()
  ps = F.softmax(torch.cat((torch.zeros(batch_size, 1).cuda(), qvalues),dim = 1),dim = 1).cuda()
  x = torch.rand(batch_size, 1).cuda()
  for i in range(batch_size):
    if ps[i, 1] > x[i, 0]: # if the agent interacts with the object/face #this is a non-greddy policy
        getreward[i] = reward[i]
    # else:              # if the agent does not interact with the object/face
    #     getreward[i] = 0
        
  return torch.sum(getreward)

save_freq = 1
for _ in range(10):     
    
    ResModel = ResNet18(64) # standard ResNet-18
    ResModel.cuda()

    # find the min of all q values for the untrained network
    min_q = 100    
    for i, data in enumerate(trainloader):
        inputs, labels, rewards = data[0].cuda(),  data[1],  data[2].cuda()
        qvalues, feature_map, second_last_features, third_last_features = ResModel(inputs)
        temp = torch.min(qvalues)
        if temp < min_q:
            min_q = temp    
    
    #get reward from the untrained network
    untrain_temp_list = []
    for i, data in enumerate(trainloader):
        inputs, labels, rewards = data[0].cuda(),  data[1],  data[2].cuda()
        qvalues, feature_map, second_last_features, third_last_features = ResModel(inputs)  
        untrain_temp = get_reward(qvalues, rewards)
        untrain_temp_list.append(untrain_temp)       
    untrain_average = sum(untrain_temp_list)/len(untrain_temp_list)
    print('reward from the untrained network: %.5f' % untrain_average)

    optimizer = optim.AdamW(ResModel.parameters(), lr = 1e-5, betas = (0.99, 0.999), weight_decay = 1e-5)
    
    ResModel.train()
    loss_memory_reward = []

    #reward only   
    for epoch in range(200):
        loss = 0
        for i, data in enumerate(trainloader):
            # get the inputs
            inputs, labels, rewards = data[0].cuda(),  data[1],  data[2].cuda()
            qvalues, feature_map, second_last_features, third_last_features = ResModel(inputs)  

            # loss
            q_loss = customloss1D(qvalues, rewards, epoch, min_q)
            train_total_loss = q_loss

            optimizer.zero_grad()
            train_total_loss.backward()

            torch.nn.utils.clip_grad_norm_(ResModel.parameters(), 0.00001)   

            optimizer.step()

            # add the mini-batch training loss to epoch loss
            loss += train_total_loss.item()
        loss = loss / len(trainloader)
        loss_memory_reward.append(loss)

        print("epoch : {}/{}, train_loss = {:.4f}".format(epoch + 1, 200, loss))

    #get reward from the trained network
    trained_temp_list = []
    for i, data in enumerate(trainloader):
        inputs, labels, rewards = data[0].cuda(),  data[1],  data[2].cuda()
        qvalues, feature_map, second_last_features, third_last_features = ResModel(inputs)  
        trained_temp = get_reward(qvalues, rewards)
        trained_temp_list.append(trained_temp)
    trained_average = sum(trained_temp_list)/len(trained_temp_list)
    print('reward from the trained network: %.5f' % trained_average)
    
    del ResModel
    torch.cuda.empty_cache()   
    