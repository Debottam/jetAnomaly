import numpy as np
import os
import h5py
from preprocessing import preprocessing 
f=h5py.File('/home/zp/gupta/jetAnomaly/utils/ttbar.h5','r')
data=f['table']
data_preprocessed=[]
for i in range(len(data)):
    if i%10000==0:
        print(i)
    data_preprocessed.append(preprocessing(data[i]))
data_preprocessed=np.vstack(data_preprocessed)
data_out=h5py.File('/home/zp/gupta/jetAnomaly/utils/ttbar_preprocessed.h5', 'w')
dset=data_out.create_dataset('table', data=data_preprocessed)
data_out.close()
f.close()
