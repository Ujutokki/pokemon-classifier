#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 21:02:18 2020

@author: dasom
"""

import sys
import os
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, regularizers
import tensorflow.keras.backend as K
from datagen import DataGenerator
print(tf.__version__)
print(sys.version)

AUTOTUNE = tf.data.experimental.AUTOTUNE

#(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()


#total_train_step = len(train_images)
#training_batch_size = 10


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
    train_datalist = []
    SearchDirectory(train_datalist,argv[1],'.jpg')
    SearchDirectory(train_datalist,argv[1],'.png')
    SearchDirectory(train_datalist,argv[1],'.gif')

    test_datalist = []
    SearchDirectory(test_datalist,argv[2],'.jpg')
    SearchDirectory(test_datalist,argv[2],'.png')
    SearchDirectory(test_datalist,argv[2],'.gif')

    train_data = DataGenerator(train_datalist,batch_size = 64,dim=(IMAGE_SIZE,IMAGE_SIZE,3),n_classes=8,shuffle=True)
    val_data = DataGenerator(test_datalist)
     

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
    model.add(layers.Conv2D(1024,(3,3),padding='same')) # output depth, kernel size
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    
    model.add(layers.DepthwiseConv2D((3,3),padding='same',strides=(2,2))) # output depth, kernel size
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(1024,(3,3),padding='same')) # output depth, kernel size
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    
    #model.add(layers.AveragePooling2D((2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(1000,activation='relu'))
    model.add(layers.Dense(9,activation='softmax'))
    model.summary()
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    #model.fit(train_data,epochs=50, validation_data=val_data)
    model.fit_generator(generator = train_data,
                        epochs=50, 
                        validation_data=val_data
                        #use_multiprocessing=True,
                        #workers=1
                        )

if __name__ == "__main__":
    if len(sys.argv)>=2:
        main(sys.argv)
    else:
        print('usage: poke.py train_data_path val_data_path')
