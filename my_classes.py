#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 11:17:47 2019

@author: dasom
"""

import numpy as np
import keras
import sys
import h5py
from keras.utils.io_utils import HDF5Matrix
import threading
import pdb
EPS = sys.float_info.epsilon
# npy data
# 0-39 logmel
# 41   label with 3 classes
# 40   label with 2 classes
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs,
                 batch_size=32, 
                 dim=(1000,40,),                  
                 n_classes=2,
                 max_len=1000, #1000 frame. 20sec
                 shuffle=True):
        
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.max_len = max_len
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size, self.dim[0]), dtype=int)
        
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            a_data = np.load(ID)
            
            #cut over max_len size
            if a_data.shape[0] > self.max_len:
                a_data = a_data[:self.max_len,:]
            
            #zero padding
            padding_data = np.pad(a_data,((0,self.dim[0]-a_data.shape[0]),(0,0)),'constant')
                                
            # Store sample
            X[i,] = padding_data[:,:40]
            # Store class
            y[i] = padding_data[:,40]            
                           

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


# npy data
# 0-39 logmel
# 41   label with 3 classes
# 40   label with 2 classes
class DataGenerator_for_hdf5(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs,
                 batch_size=32, 
                 dim=(40,),                  
                 n_classes=2,                 
                 shuffle=False):
        
        'Initialization'
        self.dim = dim        
        self.list_IDs = list_IDs
        self.n_classes = n_classes   
        self.shuffle = shuffle
        self.on_epoch_end()
        self.batch_size = batch_size
   

    def __len__(self):
        'Denotes the number of batches per epoch'
        total_data_num = 0
        for trainfile in self.list_IDs:
            total_data_num += len(HDF5Matrix(trainfile, 'target'))
        #return int(np.floor(len(self.list_IDs) / self.batch_size))
        return int(np.floor(total_data_num/self.batch_size))
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        #indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        #list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        #X, y = self.__data_generation(list_IDs_temp)        
        X, y = self.__data_generation(self.list_IDs)        
        return X, y
    

    def on_epoch_end(self):
        #'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
            
    def __data_generation(self,list_IDs): # shuffle whitin a hdf5 file

        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)
        
        
 
        for idx,hdf5_file in enumerate(self.list_IDs):           
            #with h5py.File(hdf5_file, 'r') as hf:
                #inputs = np.asarray(hf['feature'].value)
                #targets = np.asarray(hf['target'].value)
                
            inputs = HDF5Matrix(hdf5_file, 'feature')
            targets = HDF5Matrix(hdf5_file, 'target')
            
            if self.shuffle:
                indices = np.arange(len(inputs))
             
               
            for start_idx in range(0, inputs.shape[0] - self.batch_size + 1, self.batch_size):
                if self.shuffle:
                    excerpt = indices[start_idx:start_idx + self.batch_size]
                else:
                    excerpt = slice(start_idx, start_idx + self.batch_size)
                X = inputs[excerpt]
                y = targets[excerpt]
                    
                yield X, keras.utils.to_categorical(y, num_classes=self.n_classes)


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g



def Normalize(x):   
    mean_std_chunk_path = '%s_mean_std_chunk_%d' % ('NF',5) 
    x_mean = np.array(open(mean_std_chunk_path + '/arr_mean_%d.csv' % 5, 'r').read().split())
    x_std = np.array(open(mean_std_chunk_path + '/arr_std_%d.csv' % 5, 'r').read().split())
    x_std = np.array(open(mean_std_chunk_path + '/arr_std_%d.csv' % 5, 'r').read().split())
    
    x_mean=x_mean.reshape((1,-1))
    x_std=x_std.reshape((1,-1))
    print(x.shape, x_mean.shape,x_std.shape)
    print(type(x), type(x_mean))
    return (x - x_mean) / x_std



#@threadsafe_generator
def my_generator(hdf5_files, batchsize=1, shuffle=False): # hdf5
    while True:
        for idx,hdf5_file in enumerate(hdf5_files):
            #print('\nopen the no.{} hdf5 file: {}'.format(idx+1, hdf5_file))
            with h5py.File(hdf5_file, 'r') as hf:
                inputs = np.array(hf['feature'].value)
                targets = np.array(hf['target'].value)   
                 
                if shuffle:
                    indices = np.arange(inputs.shape[0])
                    np.random.shuffle(indices)
                for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
                    if shuffle:
                        excerpt = indices[start_idx:start_idx + batchsize]
                    else:
                        excerpt = slice(start_idx, start_idx + batchsize)
                    yield inputs[excerpt], targets[excerpt]

                  
