import numpy as np
import tensorflow as tf
import cv2
import io
from PIL import Image,ImageCms



def decode_img(ID,IMG_WIDTH,IMG_HEIGHT):
  #with Image.open(ID) as img:
  #with open(ID,'rb') as img:
      #img = convert_to_srgb(img)
      #img = np.array(img.read())
      img = tf.io.read_file(ID)
      # convert the compressed string to a 3D uint8 tensor
      img = tf.image.decode_jpeg(img, channels=3)
      #img = tf.image.decode_image(img, channels=3)
      #img = tf.image.decode_png(img, channels=3)
      # Use `convert_image_dtype` to convert to floats in the [0,1] range.
      img = tf.image.convert_image_dtype(img,tf.float32)
      # resize the image to the desired size.
      return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])


class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=1, dim=(64,64,3), n_channels=1,
                 n_classes=9, shuffle=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        #self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
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
        #X = np.empty((self.batch_size, *self.dim, self.n_channels))
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            #X[i,] = np.load('data/' + ID + '.npy')
            x = decode_img(ID,self.dim[0],self.dim[1])#/255.0
            X[i,] = x
            # Store class
            label = int( ID.split('/')[-1][:3])-1
            #print(ID,label)
            y[i] = label
         
            #print(ID,label)    
 
         

        return X, y   #tf.keras.utils.to_categorical(y, num_classes=self.n_classes)
