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
        image = Image.open(self.x.iloc[index]).convert('RGB')

        if self.transforms:
            image = self.transforms(image)

        label = self.y.iloc[index]
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

test_dataset = FaceDataset(txt_path = '/data/zhouabx/celebA/test_SUBSET.pkl',
                           img_dir = '/mmfs1/data/zhouabx/celebA/img_align_celeba/',
                           transforms = transform)                           
testloader = DataLoader(dataset = test_dataset, batch_size = 64, shuffle = True)


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
ResModel.cuda()

optimizer = optim.AdamW(ResModel.parameters(), lr = 1e-4, betas = (0.99, 0.999), weight_decay = 1e-4)
classification_criterion = nn.CrossEntropyLoss()

save_root = "/data/zhouabx/ResNet-18/ResNet_gender/id/test_CelebA_train_results"

save_freq = 1
ResModel.train()
loss_memory_classification = []
for epoch in range(200):
    loss = 0
    for i, data in enumerate(trainloader):
        # get the inputs
        inputs, labels = data[0].cuda(),  data[1].cuda()
        x, classification = ResModel(inputs)  

        # loss
        classification_loss = classification_criterion(classification, labels)
        train_total_loss = classification_loss 

        optimizer.zero_grad()
        train_total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(ResModel.parameters(), 0.001)   
        
        optimizer.step()

        # add the mini-batch training loss to epoch loss
        loss += train_total_loss.item()
        
    loss = loss / len(trainloader)
    loss_memory_classification.append(loss)

    print("epoch : {}/{}, train_loss = {:.4f}".format(epoch + 1, 200, loss))
    
# loss_memory.append(epoch_loss)
running_loss = 0.0
if epoch % save_freq == save_freq - 1:
    savename = f'RGB_ResNet_ID.ckp'
    save_path = os.path.join(save_root, savename)
    torch.save({
        'epoch_classification': epoch,
        'ResModel_state_dict': ResModel.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_memory_classification': loss_memory_classification,
        }, save_path)

from sklearn.metrics import confusion_matrix    
# Test
ResModel.eval()
score = []
FP_list = []
FN_list = []

for i, data in enumerate(testloader):
    # get the inputs
    inputs, labels = data[0].cuda(),  data[1].cuda()

    x, classification = ResModel(inputs)
    outputs_numpy = torch.Tensor.cpu(classification).cpu().data.numpy()
    outputs_argmax = np.argmax(outputs_numpy,axis = 1)
    labels_numpy = labels.cpu().data.numpy()
    score = np.concatenate((score,(labels_numpy == outputs_argmax).astype(int)),axis = 0)
        
    the_FP = confusion_matrix(labels_numpy, outputs_argmax, normalize = 'all')[0][1]
    the_FN = confusion_matrix(labels_numpy, outputs_argmax, normalize = 'all')[1][0]
        
    FP_list = np.concatenate((FP_list, [the_FP]), axis = 0)
    FN_list = np.concatenate((FN_list, [the_FN]), axis = 0)

meanAccuracy = sum(score)/len(score)
print("Accuracy: {:.5f}".format(meanAccuracy))

meanFP = sum(FP_list)/len(FP_list)
print("False Positive: {:.5f}".format(meanFP))

meanFN = sum(FN_list)/len(FN_list)
print("False Negative: {:.5f}".format(meanFN))
