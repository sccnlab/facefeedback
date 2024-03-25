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
import torchvision.models as models

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
        self.conv1 = nn.Conv2d(3, k, kernel_size = 3, stride = 1, padding = 1, bias = False)
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
        out = self.layer3(out)       
        out = self.layer4(out)
        return out
   
class ResNet18(nn.Module):
    def __init__(self, k):
        super(ResNet18, self).__init__()
        self.encoder = ResNetEncoder([2, 2, 2, 2], k)
        
        # id classifier 
        self.classifier = nn.Sequential(
            nn.Linear(512, 1503)
        )

    def forward(self, x):
        feature_map = self.encoder(x)
        x = F.adaptive_avg_pool2d(feature_map, (1, 1))
        x = x.view(x.size(0), -1)
        classification = self.classifier(x)
        return x, classification
ResModel = ResNet18(64)

class ResNetVAE(nn.Module):
    def __init__(self, latent_dim = 256):
        super(ResNetVAE, self).__init__()      
        # resnet = models.resnet18(pretrained = False)
        resnet = ResModel.cuda()
        self.encoder = nn.Sequential(*list(resnet.children())[:-1]) # remove the fully connected layers (fc)
        
        # fully connected part
        self.fc_mu = nn.Linear(512 * 16 * 16, latent_dim)     
        self.fc_logvar = nn.Linear(512 * 16 * 16, latent_dim)
        
        
        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, 512 * 16 * 16)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size = 4, stride = 2, padding = 1),   
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size = 4, stride = 2, padding = 1),  
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size = 4, stride = 2, padding = 1),   
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(3)
        )

    def reparameterize(self, mu, logVar):
        std = torch.exp(0.5 * logVar) #logVar = log(sigma^2) -> 0.5*logVar.exp = sigma
        eps = torch.randn_like(std)
        # reparametrisation trick
        return mu + std * eps
 
    def forward(self, x):    # encoder -> reparameterization -> decoder
        x = self.encoder(x)
        x = torch.flatten(x, start_dim = 1)      

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        z = self.reparameterize(mu, logvar)       
 
        x = self.decoder_fc(z)
        x = x.view(-1, 512, 16, 16) 
        reconstruction = self.decoder(x)
        return reconstruction, mu, logvar
    
model = ResNetVAE()
model.cuda()

optimizer = optim.AdamW(model.parameters(), lr = 1e-4, betas = (0.99, 0.999), weight_decay = 1e-5)
reconstruction_criterion = nn.MSELoss()

#Reconstruction only 
save_root = "/data/zhouabx/ResNet-18/ResNet_UNSUP/CelebA_train_results"
save_freq = 1
model.train()
loss_memory_recon = []

for epoch in range(500):
    loss = 0
    for i, data in enumerate(trainloader):
        # get the inputs
        inputs, labels, rewards = data[0].cuda(),  data[1],  data[2].cuda()
        reconstruction, mu, logVar = model(inputs)  

        # loss
        reconstruction_loss = reconstruction_criterion(reconstruction, inputs)
        KLD_loss = -0.5 * torch.sum(1 + logVar - mu.pow(2) - logVar.exp()) # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        train_total_loss = 4000 * reconstruction_loss + 0.0000001 * KLD_loss

        optimizer.zero_grad()
        train_total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.001)   
        
        optimizer.step()

        # add the mini-batch training loss to epoch loss
        loss += train_total_loss.item()
    loss = loss / len(trainloader)
    loss_memory_recon.append(loss)

    print("epoch : {}/{}, train_loss = {:.4f}".format(epoch + 1, 500, loss))
    
    
# loss_memory.append(epoch_loss)
running_loss = 0.0
if epoch % save_freq == save_freq - 1:
    savename = f'VAE_ResNet_UNSUP_001_500_epoch.ckp'
    save_path = os.path.join(save_root, savename)
    torch.save({
        'epoch_reconstruction': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_memory_reconstruction': loss_memory_recon,
        }, save_path)

    
from torch.utils.data import Dataset, DataLoader
class KDEFVersionB(Dataset):
    """KDEFVersionB dataset."""
    def __init__(self, txt_path, img_dir, transforms = None):
        df = pickle.load(open(txt_path, "rb"))
        df['path'] = df['path'].str.replace("/mmfs1/data/schwarex/neuralNetworks/datasets/KDEF", img_dir) 
        self.txt_path = txt_path
        self.y = df['id_number']
        self.x = df['path']
        self.n_samples = len(df)
        self.transforms = transforms
        
        self.mu = torch.randn(200)
        m = torch.distributions.gamma.Gamma(torch.ones(200), torch.ones(200) * 20)
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
              
    
transform = transforms.Compose([transforms.Resize((128, 128)),
                                transforms.ToTensor()])

test_dataset_B = KDEFVersionB(txt_path = '/mmfs1/data/zhouabx/KDEF/Subset_B_reset_index.pkl',
                              img_dir = '/mmfs1/data/zhouabx/KDEF/KDEF_Copy/',
                              transforms = transform)
      
testloader_B = DataLoader(dataset = test_dataset_B, batch_size = 64, shuffle = False) 


path = "/data/zhouabx/ResNet-18/ResNet_UNSUP/Reconstructions/"
model.eval()
for data in random.sample(list(testloader_B), 1):
    imgs = data[0]
    imgs = imgs.cuda()
    inputimg_testloader_B = np.transpose(imgs[0].cpu().numpy(), [1,2,0])
#save input image
    with open(path + 'VAE_inputimg_testloader_B_001_500_epoch.pkl', 'wb') as pickle_out:
        pickle.dump(inputimg_testloader_B, pickle_out)

    reconstruction, mu, logVar = model(imgs)
    outimg_testloader_B = np.transpose(reconstruction[0].cpu().detach().numpy(), [1,2,0])
#save output image
    with open(path + 'VAE_outimg_testloader_B_001_500_epoch.pkl', 'wb') as pickle_out:
        pickle.dump(np.squeeze(outimg_testloader_B), pickle_out)        
    break
    