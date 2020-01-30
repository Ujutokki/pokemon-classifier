#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 21:02:18 2020

@author: dasom
"""

import sys
import os
import random

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, regularizers
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
print(tf.__version__)
print(sys.version)
from datagen import DataGenerator

AUTOTUNE = tf.data.experimental.AUTOTUNE

IMAGE_SIZE = 64 

def SearchDirectory(ret,directory,filename):
    for t in os.listdir(directory):
        if os.path.isdir(os.path.join(directory,t)):
            searchDirectory(ret,os.path.join(directory,t),filename)
        else:
            if filename in t :
                ret.append(os.path.join(directory,t))
    return


def main(argv):
    datalist = []
    SearchDirectory(datalist,argv[1],'.jpeg')
    #SearchDirectory(datalist,argv[1],'.jpg')
    #SearchDirectory(datalist,argv[1],'.png')
    #SearchDirectory(datalist,argv[1],'.gif')

    random.seed(44)
    random.shuffle(datalist)
    train_datalist = datalist[100:]
    val_datalist = datalist[:100]

    train_data = DataGenerator(train_datalist,batch_size = 64,
                               dim=(IMAGE_SIZE,IMAGE_SIZE,3),
                               n_classes=150,
                               shuffle=True)

    val_data = DataGenerator(val_datalist)
   
    '''
    test_datalist = []
    SearchDirectory(test_datalist,argv[2],'.jpg')
    SearchDirectory(test_datalist,argv[2],'.png')
    SearchDirectory(test_datalist,argv[2],'.gif')

    val_data = DataGenerator(test_datalist)
    ''' 

    """**MOBILE NET**"""
    
    model = models.Sequential()
    model.add(layers.Conv2D(32,(3,3),padding='same',input_shape=(IMAGE_SIZE,IMAGE_SIZE,3),strides=(2,2))) # output depth, kernel size
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    
    model.add(layers.DepthwiseConv2D((3,3),padding='same')) # output depth, kernel size
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(64,(3,3),padding='same')) # output depth, kernel size
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    
    model.add(layers.DepthwiseConv2D((3,3),padding='same',strides=(2,2))) # output depth, kernel size
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(64,(3,3),padding='same')) # output depth, kernel size
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    
    model.add(layers.DepthwiseConv2D((3,3),padding='same')) # output depth, kernel size
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(128,(3,3),padding='same')) # output depth, kernel size
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    
    '''
    model.add(layers.DepthwiseConv2D((3,3),padding='same',strides=(2,2))) # output depth, kernel size
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(128,(3,3),padding='same')) # output depth, kernel size
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    
    model.add(layers.DepthwiseConv2D((3,3),padding='same')) # output depth, kernel size
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(256,(3,3),padding='same')) # output depth, kernel size
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    
    model.add(layers.DepthwiseConv2D((3,3),padding='same',strides=(2,2))) # output depth, kernel size
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(256,(3,3),padding='same')) # output depth, kernel size
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    for i in range(5):
      model.add(layers.DepthwiseConv2D((3,3),padding='same')) # output depth, kernel size
      model.add(layers.BatchNormalization())
      model.add(layers.Activation('relu'))
      model.add(layers.Conv2D(512,(3,3),padding='same')) # output depth, kernel size
      model.add(layers.BatchNormalization())
      model.add(layers.Activation('relu'))

    model.add(layers.DepthwiseConv2D((3,3),padding='same',strides=(2,2))) # output depth, kernel size
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(128,(3,3),padding='same')) # output depth, kernel size
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    '''
    
    model.add(layers.DepthwiseConv2D((3,3),padding='same',strides=(2,2))) # output depth, kernel size
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(128,(3,3),padding='same')) # output depth, kernel size
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    
    model.add(layers.AveragePooling2D((2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(50,activation='relu'))
    model.add(layers.Dense(150,activation='softmax'))
    model.summary()
    
    model.compile(optimizer=Adam(lr=1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    #model.fit(train_data,epochs=50, validation_data=val_data)
    model.fit_generator(generator = train_data,
                        epochs=50, 
                        validation_data=val_data,
                        #use_multiprocessing=True,
                        workers=32
                        )
    '''
    model.fit(generator = train_data,
              epochs=50, 
              validation_data=val_data,
              #use_multiprocessing=True,
              workers=32
              )

    '''

if __name__ == "__main__":
    if len(sys.argv)>=1:
        main(sys.argv)
    else:
        print('usage: poke.py train_data_path val_data_path')
