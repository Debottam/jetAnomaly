import numpy as np
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import h5py
import pickle
import pandas
import matplotlib.pyplot as plt
#import deepdish.io as io
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.utils import plot_model
from preprocessing import preprocessing 
f=h5py.File('/lcg/storage13/atlas/gupta/stealth_boson/Inter_Zp2200_S80A15.h5','r')
data=f['table']
data_preprocessed=[]
for i in range(len(data)):
    if i%10000==0:
        print(i)
    data_preprocessed.append(preprocessing(data[i]))
data_preprocessed=np.vstack(data_preprocessed)
data_out=h5py.File('/lcg/storage13/atlas/gupta/stealth_boson/Inter_Zp2200_S80A15_preprocessed.h5', 'w')
dset=data_out.create_dataset('table', data=data_preprocessed)
data_out.close()
f.close()
