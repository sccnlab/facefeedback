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

os.environ["CUDA_VISIBLE_DEVICES"] = "1, 3"

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

#three denseblocks w/ reduced layer classifier & reduced two avg_pool2d & add weight_decay
class NetDense(nn.Module):
    def __init__(self, k):
        super(NetDense, self).__init__()
        self.k = k
        #(128, 128, 3)
        self.bn1 = nn.BatchNorm2d(3) #,track_running_stats=False)
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
        
        
        ## classifier with reduced layer
        self.classifier = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1503) #1503 ids
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
        #x7 = F.avg_pool2d(x7, 2)  
       
        # third dense block
        x8 = F.relu(self.encConv8(self.bn8(x7)))
        x = torch.cat((x7, x8), 1)
        x9 = F.relu(self.encConv9(self.bn9(x)))
        x = torch.cat((x7, x8, x9), 1)
        
        x10 = F.relu(self.encConv10(self.bn10(x)))         
        #x10 = F.avg_pool2d(x10, 2)        
                    
        x = self.bn11(x10)
                
        x = x.view(-1, self.k * 32 * 32)
        return self.encoder_mu(x), self.encoder_logVar(x)       
    
    
    def reparameterize(self, mu, logVar):
        std = torch.exp(0.5 * logVar) #logVar = log(sigma^2) -> 0.5*logVar.exp = sigma
        eps = torch.randn_like(std)
        # reparametrisation trick
        return mu + std * eps
 
    
    def classify(self, z):
        return self.classifier(z)

    def forward(self, x):  # encoder -> reparameterization -> decoder
        mu, logVar = self.encoder(x)
        z = self.reparameterize(mu, logVar)
        classification = self.classifier(z)
        #reconstruction = self.decoder(z)
        return classification, mu, logVar

netDense = NetDense(64) #k = 64
netDense.cuda()  

#optimizer = optim.Adam(netDense.parameters(), lr = 1e-4, betas = (0.99, 0.999))

optimizer = optim.AdamW(netDense.parameters(), lr = 1e-4, betas = (0.99, 0.999), weight_decay = 1e-5)
classification_criterion = nn.CrossEntropyLoss()

save_root = "/data/zhouabx/KDEF/KDEF_RDM_new/VAE_classification/CelebA_train_results"
save_freq = 1
netDense.train()
loss_memory_classification = []

for epoch in range(50):
    loss = 0
    for i, data in enumerate(trainloader):
        # get the inputs
        inputs, labels = data[0].cuda(),  data[1].cuda()
        classification, mu, logVar = netDense(inputs)  

        # loss
        classification_loss = classification_criterion(classification, labels)
        KLD_loss = -0.5 * torch.sum(1 + logVar - mu.pow(2) - logVar.exp()) # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        train_total_loss = classification_loss + 0.0000000000001 * KLD_loss

        optimizer.zero_grad()
        train_total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(netDense.parameters(), 0.00001)   
        
        optimizer.step()

        # add the mini-batch training loss to epoch loss
        loss += train_total_loss.item()
        
    loss = loss / len(trainloader)
    loss_memory_classification.append(loss)

    print("epoch : {}/{}, train_loss = {:.4f}".format(epoch + 1, 50, loss))

# loss_memory.append(epoch_loss)
running_loss = 0.0
if epoch % save_freq == save_freq - 1:
    savename = f'only_classification.ckp'
    save_path = os.path.join(save_root, savename)
    torch.save({
        'epoch_classification': epoch,
        'netDense_state_dict': netDense.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_memory_classification': loss_memory_classification,
        }, save_path)


