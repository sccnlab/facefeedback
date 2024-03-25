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
Dataset_A = pickle.load(open('/mmfs1/data/zhouabx/KDEF/data_A_with_id_expression_direction.pkl', "rb")) ## (note that this is described as dataset B in the paper, and vice versa the dataset that is described as dataset A in the paper is named dataset B in the code)
image_list_A = []
for idx, row in Dataset_A.iterrows():
    image = Image.open(Dataset_A['path'][idx])
    image_list_A.append(image)

image_array_list_A = []
for image in image_list_A:
    image_array = np.array(image)
    image_array_list_A.append(image_array)
    
flattened_image_array_list_A = [arr.flatten() for arr in image_array_list_A]

Dataset_A['flattened_list_A'] = flattened_image_array_list_A
Dataset_A_with_flattened_image = Dataset_A  

#average the L and R     
Dataset_A_with_flattened_image["new_direction"] = np.where((Dataset_A_with_flattened_image["direction"].values == "HR") | 
                                                           (Dataset_A_with_flattened_image["direction"] == "HL"), "H", 
                                                           np.where((Dataset_A_with_flattened_image["direction"].values == "FR") | 
                                                                    (Dataset_A_with_flattened_image["direction"] == "FL"), "F", 
                                                                    Dataset_A_with_flattened_image["direction"].values))        
         

group_index = Dataset_A_with_flattened_image.agg('{0[expression]} and {0[new_direction]} and {0[id]}'.format, axis = 1)

ids = np.unique(group_index)
g_mean = np.array([np.mean(np.array(flattened_image_array_list_A)[group_index == i, 0:], axis = 0) for i in ids])

df_LR_ave = pd.DataFrame(data = np.array(list(map(lambda x : np.str.split(x, sep = " and "), ids))))
df_LR_ave.columns = ["expression", "direction", "id"]
df_LR_ave['flattened_list_A'] = g_mean.tolist()

# #save df_LR_ave
# path = "/data/zhouabx/iEEG_rdms/Pixel_Similarity/Result/"
# with open(path + 'df_LR_ave.pkl', 'wb') as pickle_out:
#      pickle.dump(df_LR_ave, pickle_out)
                
#label everything
dictionary_emotion_A = {'AF' : 0, 'AN' : 1, 'HA' : 2, 'NE' : 3, 'SA' : 4}
df_LR_ave['emotionlabel_A'] = df_LR_ave['expression'].map(dictionary_emotion_A)

dictionary_view_A = {'F' : 0, 'H' : 1, 'S' : 2}
df_LR_ave['viewlabel_A'] = df_LR_ave['direction'].map(dictionary_view_A)

dictionary_id_A = {'AF01' : 0, 'AF02' : 1, 'AF03' : 2, 'AF04' : 3, 'AF06' : 4, 'AF07' : 5, 'AF09' : 6, 'AF13' : 7, 'AF14' : 8, 'AF16' : 9,
 'AF20' : 10, 'AF21' : 11, 'AF22' : 12, 'AF24' : 13, 'AF25': 14, 'AF28' : 15, 'AF29' : 16, 'AF30' : 17, 'AF32' : 18, 
 'AF33' : 19, 'AM01' : 20, 'AM05' : 21, 'AM06' : 22, 'AM08' : 23, 'AM09' : 24, 'AM10' : 25, 'AM11' : 26, 'AM13' : 27,
 'AM14' : 28, 'AM15' : 29, 'AM16' : 30, 'AM22' : 31, 'AM23' : 32, 'AM24' : 33, 'AM25' : 34, 'AM28' : 35, 'AM29' : 36,
 'AM30' : 37, 'AM31' : 38, 'AM35' : 39}
df_LR_ave['idlabel_A'] = df_LR_ave['id'].map(dictionary_id_A)

#order everything
New_Data_A = df_LR_ave.sort_values(by = ['emotionlabel_A', 'viewlabel_A', 'idlabel_A'])
#reset the index 
New_Data_A_with_flattened_image = New_Data_A.reset_index(drop = True)  


# create array features
array_features_A = [New_Data_A_with_flattened_image['flattened_list_A'][0]]
for i in range(1, New_Data_A_with_flattened_image.shape[0]):
    array_features_A = np.concatenate((array_features_A, [New_Data_A_with_flattened_image['flattened_list_A'][i]]), axis = 0)

# substract baseline
average_vector = np.mean(array_features_A, axis = 0)
centered_data = array_features_A - average_vector

# attach centered_feature
New_Data_A_with_flattened_image['centered_feature'] = [centered_data[i] for i in range(len(centered_data))]


l_A = [None] * 600
for _ in range (len(New_Data_A_with_flattened_image)):
    l_A [New_Data_A_with_flattened_image.iloc[_]['idlabel_A'] + 
         40 * New_Data_A_with_flattened_image.iloc[_]['viewlabel_A'] + 
         40 * 3 * New_Data_A_with_flattened_image.iloc[_]['emotionlabel_A']] = New_Data_A_with_flattened_image.iloc[_]['centered_feature']      

RDM_A = 1 - np.corrcoef(l_A)      
    
#save RDM value
path = "/data/zhouabx/KDEF/KDEF_RDM_new/2023_Oct_RDMs/Pixel_Similarity/RDM_Results/"
with open(path + 'RDM_value_A.pkl', 'wb') as pickle_out:
     pickle.dump(RDM_A, pickle_out)     
        
