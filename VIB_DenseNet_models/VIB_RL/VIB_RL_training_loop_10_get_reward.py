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
#import pickle5 as pickle

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"


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

    
transform = transforms.Compose([transforms.CenterCrop(178),  #original image size = (178, 218)
                                transforms.Resize(128),
                                transforms.ToTensor()])       
                        

train_dataset = FaceDataset(txt_path = '/data/zhouabx/celebA/train_SUBSET.pkl',
                            img_dir = '/mmfs1/data/zhouabx/celebA/img_align_celeba/',
                            transforms = transform)

trainloader = DataLoader(dataset = train_dataset, batch_size = 64, shuffle = True)


class MultitaskNetDense(nn.Module):
    def __init__(self, k):
        super(MultitaskNetDense, self).__init__()
        self.k = k
        #(128, 128, 3)
        self.bn1 = nn.BatchNorm2d(3) 
        self.encConv1 = nn.Conv2d(3, k, kernel_size = 4, stride = 2, padding = 1) # -> 16 x 64 x 64
        self.bn2 = nn.BatchNorm2d(k)

        # first dense block
        self.encConv2 = nn.Conv2d(k, k, kernel_size = 3, stride = 1, padding = 1) # -> 16 x 64 x 64
        self.bn3 = nn.BatchNorm2d(k * 2)
        self.encConv3 = nn.Conv2d(k * 2, k, kernel_size = 3, stride = 1, padding = 1) # -> 16 x 64 x 64
        self.bn4 = nn.BatchNorm2d(k * 3)
        self.encConv4 = nn.Conv2d(k * 3, k, kernel_size = 3, stride = 1, padding = 1) # -> 16 x 64 x 64
        self.bn5 = nn.BatchNorm2d(k)

        # second dense block
        self.encConv5 = nn.Conv2d(k, k, kernel_size = 3, stride = 1, padding = 1) # -> 16 x 64 x 64
        self.bn6 = nn.BatchNorm2d(k * 2)
        self.encConv6 = nn.Conv2d(k * 2, k, kernel_size = 3, stride = 1, padding = 1) # -> 16 x 64 x 64
        self.bn7 = nn.BatchNorm2d(k * 3)
        self.encConv7 = nn.Conv2d(k * 3, k, kernel_size = 3, stride = 1, padding = 1) # -> 16 x 64 x 64
        self.bn8 = nn.BatchNorm2d(k)
        
        # third dense block
        self.encConv8 = nn.Conv2d(k, k, kernel_size = 3, stride = 1, padding = 1) # -> 16 x 64 x 64
        self.bn9 = nn.BatchNorm2d(k * 2)
        self.encConv9 = nn.Conv2d(k * 2, k, kernel_size = 3, stride = 1, padding = 1) # -> 16 x 64 x 64
        self.bn10 = nn.BatchNorm2d(k * 3)
        self.encConv10 = nn.Conv2d(k * 3, k, kernel_size = 3, stride = 1, padding = 1) # -> 16 x 64 x 64
        self.bn11 = nn.BatchNorm2d(k) 
        
        # fully connected part
        self.encoder_mu = nn.Linear(k * 32 * 32, 2048)  
        self.encoder_logVar = nn.Linear (k * 32 * 32, 2048)         
        
        #add dropout layer
        self.dropout = nn.Dropout2d(0.25) 
        
        self.decFC1 = nn.Linear(2048, k * 32 * 32)
        self.bn12 = nn.BatchNorm2d(k)
        
        self.decConv1 = nn.ConvTranspose2d(k, k, kernel_size = 3, stride = 1, padding = 1) 
        self.bn13 = nn.BatchNorm2d(k * 2)
        self.decConv2 = nn.ConvTranspose2d(k * 2, k, kernel_size = 3, stride = 1, padding = 1) 
        self.bn14 = nn.BatchNorm2d(k * 3)
        self.decConv3 = nn.ConvTranspose2d(k * 3, k, kernel_size = 3, stride = 1, padding = 1) 
        self.bn15 = nn.BatchNorm2d(k)
 
        self.decConv4 = nn.ConvTranspose2d(k, k, kernel_size = 3, stride = 1, padding = 1) 
        self.bn16 = nn.BatchNorm2d(k * 2)
        self.decConv5 = nn.ConvTranspose2d(k * 2, k, kernel_size = 3, stride = 1, padding = 1) 
        self.bn17 = nn.BatchNorm2d(k * 3)
        self.decConv6 = nn.ConvTranspose2d(k * 3, k, kernel_size = 3, stride = 1, padding = 1) 
        self.bn18 = nn.BatchNorm2d(k)
        
        self.decConv7 = nn.ConvTranspose2d(k, k, kernel_size = 3, stride = 1, padding = 1) 
        self.bn19 = nn.BatchNorm2d(k * 2)
        self.decConv8 = nn.ConvTranspose2d(k * 2, k, kernel_size = 3, stride = 1, padding = 1) 
        self.bn20 = nn.BatchNorm2d(k * 3)
        self.decConv9 = nn.ConvTranspose2d(k * 3, k, kernel_size = 3, stride = 1, padding = 1) 
        self.bn21 = nn.BatchNorm2d(k)
        
        self.decConv10 = nn.ConvTranspose2d(k, 3, kernel_size = 4, stride = 2, padding = 1) 
        self.bn22 = nn.BatchNorm2d(3)
                
            
        ## q network
        self.qnetwork = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )  
        
        
    def encoder(self, x):
        # first dense block
        x1 = F.relu(self.encConv1(self.bn1(x)))
        x2 = F.relu(self.encConv2(self.bn2(x1)))       
        x = torch.cat((x1, x2), 1) 
        
        x3 = F.relu(self.encConv3(self.bn3(x)))
        x = torch.cat((x1, x2, x3), 1)
        
        x4 = F.relu(self.encConv4(self.bn4(x)))
        x4 = F.avg_pool2d(x4, 2)
        
        # second dense block
        x5 = F.relu(self.encConv5(self.bn5(x4)))
        x = torch.cat((x4, x5), 1)
        x6 = F.relu(self.encConv6(self.bn6(x)))
        x = torch.cat((x4, x5, x6), 1)
        
        x7 = F.relu(self.encConv7(self.bn7(x)))          
       
        # third dense block
        x8 = F.relu(self.encConv8(self.bn8(x7)))
        x = torch.cat((x7, x8), 1)
        x9 = F.relu(self.encConv9(self.bn9(x)))
        x = torch.cat((x7, x8, x9), 1)
        
        x10 = F.relu(self.encConv10(self.bn10(x)))                
                    
        x = self.bn11(x10)
                
        x = x.view(-1, self.k * 32 * 32)
        x = self.dropout(x)
        return self.encoder_mu(x), self.encoder_logVar(x)                   
 

    def reparameterize(self, mu, logVar):
        std = torch.exp(0.5 * logVar) #logVar = log(sigma^2) -> 0.5*logVar.exp = sigma
        eps = torch.randn_like(std)
        # reparametrisation trick
        return mu + std * eps
    
    
    def decoder(self, z):
        # first dense block
        x = F.relu(self.decFC1(z))
        
        x11 = x.view(-1, self.k, 32, 32)      
        x12 = F.relu(self.decConv1(self.bn12(x11)))             
        x = torch.cat((x11, x12), 1)         
        
        x13 = F.relu(self.decConv2(self.bn13(x)))
        x = torch.cat((x11, x12, x13), 1)  
        
        x14 = F.relu(self.decConv3(self.bn14(x)))
        
        # second dense block
        x15 = F.relu(self.decConv4(self.bn15(x14)))
        x = torch.cat((x14, x15), 1) 
        
        x16 = F.relu(self.decConv5(self.bn16(x)))
        x = torch.cat((x14, x15, x16), 1)  
        
        x17 = F.relu(self.decConv6(self.bn17(x)))
        
        # third dense block
        x18 = F.relu(self.decConv7(self.bn18(x17)))
        x = torch.cat((x17, x18), 1) 
        
        x19 = F.relu(self.decConv8(self.bn19(x)))
        x = torch.cat((x17, x18, x19), 1) 
        
        x20 = F.relu(self.decConv9(self.bn20(x)))        
        x20 = F.interpolate(x20, [64, 64])
              
        x21 = F.relu(self.decConv10(self.bn21(x20)))
        x = self.bn22(x21)
        return x
    
    def q(self, z):
        return self.qnetwork(z)
 
    def forward(self, x):  # encoder -> reparameterization -> decoder
        mu, logVar = self.encoder(x)
        z = self.reparameterize(mu, logVar)
        qvalues = self.q(z)
        reconstruction = self.decoder(z)
        return qvalues, reconstruction, mu, logVar
    
    
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
    
    model = MultitaskNetDense(64)
    model.cuda()
    
    # find the min of all q values for the untrained network
    min_q = 100    
    for i, data in enumerate(trainloader):
        inputs, labels, rewards = data[0].cuda(),  data[1],  data[2].cuda()
        qvalues, reconstruction, mu, logVar = model(inputs)
        temp = torch.min(qvalues)
        if temp < min_q:
            min_q = temp    
    
    #get reward from the untrained network
    untrain_temp_list = []
    for i, data in enumerate(trainloader):
        inputs, labels, rewards = data[0].cuda(),  data[1],  data[2].cuda()
        qvalues, reconstruction, mu, logVar = model(inputs)  
        untrain_temp = get_reward(qvalues, rewards)
        untrain_temp_list.append(untrain_temp)       
    untrain_average = sum(untrain_temp_list)/len(untrain_temp_list)
    print('reward from the untrained network: %.5f' % untrain_average)
        
    optimizer = optim.AdamW(model.parameters(), lr = 1e-4, betas = (0.99, 0.999), weight_decay = 1e-5)
    reconstruction_criterion = nn.MSELoss()
    
    model.train()
    loss_memory_reward = []
    
    #reward only   
    for epoch in range(50):
        loss = 0
        for i, data in enumerate(trainloader):
            # get the inputs
            inputs, labels, rewards = data[0].cuda(),  data[1],  data[2].cuda()
            qvalues, reconstruction, mu, logVar = model(inputs)  
            
            # loss
            q_loss = customloss1D(qvalues, rewards, epoch, min_q)
            reconstruction_loss = reconstruction_criterion(reconstruction, inputs)
            KLD_loss = -0.5 * torch.sum(1 + logVar - mu.pow(2) - logVar.exp()) # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            train_total_loss = 1.5 * q_loss + 0 * reconstruction_loss + 0.0000001 * KLD_loss
            
            optimizer.zero_grad()
            train_total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.00001)   
            optimizer.step()
            
            loss += train_total_loss.item()
        loss = loss / len(trainloader)
        loss_memory_reward.append(loss)
        
        print("epoch : {}/{}, train_loss = {:.4f}".format(epoch + 1, 50, loss))
        
        
    #get reward from the trained network
    trained_temp_list = []
    for i, data in enumerate(trainloader):
        inputs, labels, rewards = data[0].cuda(),  data[1],  data[2].cuda()
        qvalues, reconstruction, mu, logVar = model(inputs)  
        trained_temp = get_reward(qvalues, rewards)
        trained_temp_list.append(trained_temp)
    trained_average = sum(trained_temp_list)/len(trained_temp_list)
    print('reward from the trained network: %.5f' % trained_average)
    
    del model
    torch.cuda.empty_cache()        
   
            
     
