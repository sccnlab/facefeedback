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
import scipy.io


import scipy.stats as stats
Dataset_B = pickle.load(open('/mmfs1/data/zhouabx/KDEF/data_B_with_id_expression_direction.pkl', "rb"))  ## (note that this is described as dataset A in the paper, and vice versa the dataset that is described as dataset B in the paper is named dataset A in the code)

image_list_B = []
for idx, row in Dataset_B.iterrows():
    image = Image.open(Dataset_B['path'][idx])
    image_list_B.append(image)

image_array_list_B = []
for image in image_list_B:
    image_array = np.array(image)
    image_array_list_B.append(image_array)

flattened_image_array_list_B = [arr.flatten() for arr in image_array_list_B]


#label everything
dictionary_emotion = {'AF' : 0, 'AN' : 1, 'HA' : 2, 'NE' : 3, 'SA' : 4}
Dataset_B['emotionlabel'] = Dataset_B['expression'].map(dictionary_emotion)

dictionary_view = {'FL' : 0, 'FR' : 1, 'HL' : 2, 'HR' : 3, 'S' : 4}
Dataset_B['viewlabel'] = Dataset_B['direction'].map(dictionary_view)

dictionary_id = {'AF01' : 0, 'AF02' : 1, 'AF03' : 2, 'AF04' : 3, 'AM08' : 4, 'AM10' : 5, 'AM11' : 6, 'AM25': 7}
Dataset_B['idlabel'] = Dataset_B['id'].map(dictionary_id)

#add feature to the dataset
Dataset_B['flattened_list_B'] = flattened_image_array_list_B

#order everything
New_Data_B = Dataset_B.sort_values(by = ['emotionlabel', 'viewlabel', 'idlabel'])
#reset the index and save the dataframe in New_Data_B_with_feature
Dataset_B_with_flattened_image = New_Data_B

# create array features
array_features_B = [Dataset_B_with_flattened_image['flattened_list_B'][0]]
for i in range(1, Dataset_B_with_flattened_image.shape[0]):
    array_features_B = np.concatenate((array_features_B, [Dataset_B_with_flattened_image['flattened_list_B'][i]]), axis = 0)

# substract baseline
average_vector = np.mean(array_features_B, axis = 0)
centered_data = array_features_B - average_vector

# attach centered_feature
Dataset_B_with_flattened_image['centered_feature'] = [centered_data[i] for i in range(len(centered_data))]

# create RDM
l = [None] * 200
for _ in range (len(Dataset_B_with_flattened_image)):
    l [Dataset_B_with_flattened_image.iloc[_]['idlabel'] + 
       8 * Dataset_B_with_flattened_image.iloc[_]['viewlabel'] + 
       8 * 5 * Dataset_B_with_flattened_image.iloc[_]['emotionlabel']] = Dataset_B_with_flattened_image.iloc[_]['centered_feature']

RDM_B = 1 - np.corrcoef(l)

#save RDM value
path = "/data/zhouabx/KDEF/KDEF_RDM_new/2023_Oct_RDMs/Pixel_Similarity/RDM_Results/"
with open(path + 'RDM_value_B.pkl', 'wb') as pickle_out:
     pickle.dump(RDM_B, pickle_out)     
        
