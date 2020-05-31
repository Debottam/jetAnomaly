#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import h5py
import pickle
import pandas
import matplotlib.pyplot as plt
#import deepdish.io as io
import tensorflow as tf
from keras.models import Model,Sequential
from keras.layers import Input, Dense, Dropout
from keras.utils import plot_model
from keras.models import load_model
from sklearn.preprocessing import scale, normalize
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from collections import Counter
import sys


# In[2]:


encoder = load_model('/home/zp/gupta/jetAnomaly/utils/VAE-FCN-model-d80-UdeMdata-cycannealing-eps1_encoder.h5')


# In[11]:


f_encoded_train=h5py.File('/lcg/storage13/atlas/gupta/encoded_x_train_mid.h5','r')
f_encoded_test = h5py.File('/lcg/storage13/atlas/gupta/encoded_x_test_mid.h5','r')


# In[13]:


encoded_x_train=f_encoded_train['table']
encoded_x_test=f_encoded_test['table']


# In[14]:


level = int(sys.argv[1])
print("level: ",level)
encoded_x_train = encoded_x_train[-level]
encoded_x_test = encoded_x_test[-level]


# In[15]:


hash_size = int(sys.argv[2])
print("hash_size: ",hash_size)
sample_size, vec_size = encoded_x_train.shape
np.random.seed(0)
projections = np.random.randn(hash_size, vec_size)
hashed_encoded = list()
for i in range(sample_size):
    bools = ''
    for j in range(hash_size):
        bool = (np.dot(encoded_x_train[i], projections[j].T) > 0).astype('int')
        #print(bool)
        bools += str(bool)
        #print(bools)
    hashed_encoded.append(bools)
len(hashed_encoded)


# In[16]:


def getDuplicatesWithInfo(hashed_encoded):
    ''' Get duplicate element in a list along with thier indices in list
     and frequency count'''
    dictOfElems = dict()
    index = 0
    # Iterate over each element in list and keep track of index
    for elem in hashed_encoded:
        # If element exists in dict then keep its index in lisr & increment its frequency
        if elem in dictOfElems:
            dictOfElems[elem][0] += 1
            dictOfElems[elem][1].append(index)
        else:
            # Add a new entry in dictionary 
            dictOfElems[elem] = [1, [index]]
            #dictOfElems[elem] = 1
        index += 1    
 
    dictOfElems = { key:value for key, value in dictOfElems.items() }
    return dictOfElems


# In[17]:


dictOfElems = getDuplicatesWithInfo(hashed_encoded)
listHash = list()
listOther = list()
for key, value in dictOfElems.items():
    listHash.append(key)
    listOther.append(value)


# In[18]:


listHash = np.array(listHash)
listOther = np.array(listOther)
bkg_train_freq = listOther[:,0]


# In[19]:


bkg_train_freq = list(bkg_train_freq)


# In[20]:


tableSize= len(listHash)
sample_size, vec_size = encoded_x_test.shape
hash_truth = list()
bkgCount = list()
count = 0
for i in range(sample_size):
    bools = ''
    a = 1
    for j in range(hash_size):
        bool = (np.dot(encoded_x_test[i], projections[j].T) > 0).astype('int')
        #print(bool)
        bools += str(bool)
        #print(bools)
    for k in range(tableSize):
        if bools == listHash[k]:
            bkgCount.append(k)
            a=0
            break
    if(a==1):
        count+=1
    hash_truth.append(a)
print(count)


# In[21]:


bkg = Counter(bkgCount)
bkga=sorted(bkg.items())
bkg_test_freq = [x[1] for x in bkga]


# In[22]:


filename_1 = '/lcg/storage13/atlas/gupta/bkg_train_freq'+sys.argv[1]+sys.argv[2]+'.h5'
h5f_train_freq = h5py.File(filename_1, 'w')
h5f_train_freq.create_dataset('table', data=bkg_train_freq, compression="gzip")
h5f_train_freq.close()


# In[23]:


filename_2 = '/lcg/storage13/atlas/gupta/bkg_test_freq'+sys.argv[1]+sys.argv[2]+'.h5'
h5f_test_freq = h5py.File(filename_2, 'w')
h5f_test_freq.create_dataset('table', data=bkg_test_freq, compression="gzip")
h5f_test_freq.close()


# In[24]:


filename_3 = '/lcg/storage13/atlas/gupta/hash_truth'+sys.argv[1]+sys.argv[2]+'.h5'
h5f_hash_truth = h5py.File(filename_3, 'w')
h5f_hash_truth.create_dataset('table', data=hash_truth, compression="gzip")
h5f_hash_truth.close()


# In[25]:


f_encoded_train.close()
f_encoded_test.close()


# In[ ]:




