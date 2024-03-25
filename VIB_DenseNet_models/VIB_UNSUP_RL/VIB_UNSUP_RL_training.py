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
    
model = MultitaskNetDense(64)
model.cuda()

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

# find the min of all q values for the untrained network
min_q = 100
for i, data in enumerate(trainloader):
    inputs, labels, rewards = data[0].cuda(),  data[1],  data[2].cuda()
    qvalues, reconstruction, mu, logVar = model(inputs)
    temp = torch.min(qvalues)
    if temp < min_q:
        min_q = temp


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

#reconstruction + reward 
save_root = "/data/zhouabx/KDEF/KDEF_RDM_new/reward+recon/Combined/CelebA_train_results"

save_freq = 1
model.train()
loss_memory_recon_reward = []

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
        train_total_loss = 1.5 * q_loss + 4000 * reconstruction_loss + 0.0000001 * KLD_loss

        optimizer.zero_grad()
        train_total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.00001)   
        
        optimizer.step()

        # add the mini-batch training loss to epoch loss
        loss += train_total_loss.item()
    loss = loss / len(trainloader)
    loss_memory_recon_reward.append(loss)

    print("epoch : {}/{}, train_loss = {:.4f}".format(epoch + 1, 50, loss))

# loss_memory.append(epoch_loss)
running_loss = 0.0
if epoch % save_freq == save_freq - 1:
    savename = f'recon_reward.ckp'
    save_path = os.path.join(save_root, savename)
    torch.save({
        'epoch_recon_reward': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_memory_recon_reward': loss_memory_recon_reward,
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
        m = torch.distributions.gamma.Gamma(torch.ones(200), torch.ones(200)*20)
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
      
testloader_B = DataLoader(dataset = test_dataset_B, batch_size = 128, shuffle = False) 

path = "/data/zhouabx/KDEF/KDEF_RDM_new/reward+recon/Combined/Reconstructions/"
model.eval()
for data in random.sample(list(testloader_B), 1):
    imgs = data[0]
    imgs = imgs.cuda()
    inputimg_testloader_B = np.transpose(imgs[0].cpu().numpy(), [1,2,0])
#save input image
    with open(path + 'inputimg_testloader_B.pkl', 'wb') as pickle_out:
        pickle.dump(inputimg_testloader_B, pickle_out)

    qvalues, reconstruction, mu, logVar = model(imgs)
    outimg_testloader_B = np.transpose(reconstruction[0].cpu().detach().numpy(), [1,2,0])
#save output image
    with open(path + 'outimg_testloader_B.pkl', 'wb') as pickle_out:
        pickle.dump(np.squeeze(outimg_testloader_B), pickle_out)
        
    break
          
    
#get reward from the trained network
trained_temp_list = []
for i, data in enumerate(trainloader):
    inputs, labels, rewards = data[0].cuda(),  data[1],  data[2].cuda()
    qvalues, reconstruction, mu, logVar = model(inputs)  
    trained_temp = get_reward(qvalues, rewards)
    trained_temp_list.append(trained_temp)
trained_average = sum(trained_temp_list)/len(trained_temp_list)
print('reward from the trained network: %.5f' % trained_average)
   

# function for getting true reward rdm    
from random import sample
def RDM(df):
    df_new = df[df['new_label'].isin(sample(range(df.new_label.unique().shape[0]), 10))].copy().reset_index(drop = True)
    df_new['re_label'] = [sorted(df_new['new_label'].unique()).index(x) for x in df_new['new_label']]
    
    class FaceDataset_10ids(Dataset):
        def __init__(self, txt_path, img_dir, transforms = None):
            df = df_new
            df['img'] = df['img'].str.replace("/mmfs1/data/schwarex/neuralNetworks/identity/datasets/img_align_celeba",
                                              img_dir)
            self.txt_path = txt_path
            self.y = df['re_label']
            self.x = df['img']
            self.n_samples = len(df)
            self.transforms = transforms
            
            self.mu = torch.randn(10)
            m = torch.distributions.gamma.Gamma(torch.ones(10), torch.ones(10) * 20)
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
        
    transform_train = transforms.Compose([transforms.CenterCrop(178),
                                          transforms.Resize(128),
                                          transforms.ToTensor()])
    
    train_dataset_10ids = FaceDataset_10ids(txt_path = df_new,
                                            img_dir = '/mmfs1/data/zhouabx/celebA/img_align_celeba/',
                                            transforms = transform_train) 
        
    # Get RDM based on true rewards
    counter = torch.zeros(10)
    allmus = [ [] for _ in range(10) ]
    allrewards = [ [] for _ in range(10) ]
    
    for i in range(train_dataset_10ids.__len__()):
        image, label, reward = train_dataset_10ids.__getitem__(i)
        if counter[label] < 10:
            mu, sigma = train_dataset_10ids.mu, train_dataset_10ids.sigma
            allmus[label].append(mu.detach().cpu())
            counter[label] = counter[label] + 1
            allrewards[label].append(reward.cpu())
        if sum(counter) == 100:
            break
            
    flat_allmus = [item[0] for sublist in allmus for item in sublist]
    flat_allrewards = [item[0] for sublist in allrewards for item in sublist]
    
    # true reward RDM
    nimages = len(flat_allrewards)
    RDM_true_reward = np.zeros((nimages,nimages))
    for i in range(nimages):
        for j in range(nimages):
            RDM_true_reward[i,j] = np.abs(flat_allrewards[i] - flat_allrewards[j])
        
    # similarity matrix - randomly pick 10 images from the selected 10 ids
    counter = torch.zeros(10)
    allmus = [ [] for _ in range(10) ]    
    for i in range(train_dataset_10ids.__len__()):
        image, label, reward = train_dataset_10ids.__getitem__(i)
        if counter[label] < 10:
            mu,sigma = model.encoder(image.unsqueeze(0).cuda())
            allmus[label].append(mu.detach().cpu())
            counter[label] = counter[label] + 1
        if sum(counter) == 100:
            break
            
    flat_allmus = [item[0] for sublist in allmus for item in sublist]
    flat_allrewards = [item[0] for sublist in allrewards for item in sublist]
    
    # recon_reward RDM
    nimages = len(flat_allrewards)
    modelRDM_recon_reward = np.zeros((nimages,nimages))
    for i in range(nimages):
        for j in range(nimages):
            modelRDM_recon_reward[i,j] = 1 - stats.pearsonr(flat_allmus[i], flat_allmus[j])[0]
       
    #true reward values & kendalltau values
    v_RDM_true_reward = RDM_true_reward[np.triu_indices(RDM_true_reward.shape[0], 1)]
    v_modelRDM_recon_reward = modelRDM_recon_reward[np.triu_indices(modelRDM_recon_reward.shape[0], 1)]
    kendalltau_TrueReward_combined = stats.kendalltau(v_RDM_true_reward, v_modelRDM_recon_reward)
   
    res = {"RDM_true_reward": RDM_true_reward,
           "Recon_Reward": modelRDM_recon_reward,
           "KT_TrueReward_combined": kendalltau_TrueReward_combined}    
    return res  

df = pickle.load(open('/data/zhouabx/celebA/train_SUBSET.pkl', "rb"))

RDM_true_reward = []
Recon_Reward = []
KT_TrueReward_combined = []
for _ in range(10):   
    RDM_true_reward.append(RDM(df)['RDM_true_reward'])
    Recon_Reward.append(RDM(df)['Recon_Reward'])
    KT_TrueReward_combined.append(RDM(df)['KT_TrueReward_combined'])

#save true reward values & kendalltau results 
path = "/data/zhouabx/KDEF/KDEF_RDM_new/reward+recon/Combined/True_reward_results/"
with open(path + 'RDM_true_reward_values.pkl', 'wb') as pickle_out:
    pickle.dump(RDM_true_reward, pickle_out)   

path = "/data/zhouabx/KDEF/KDEF_RDM_new/reward+recon/Combined/True_reward_results/"
with open(path + 'Recon_Reward_values.pkl', 'wb') as pickle_out:
    pickle.dump(Recon_Reward, pickle_out) 

path = "/data/zhouabx/KDEF/KDEF_RDM_new/reward+recon/Combined/True_reward_results/"
with open(path + 'KT_TrueReward_combined.pkl', 'wb') as pickle_out:
    pickle.dump(KT_TrueReward_combined, pickle_out)     
 