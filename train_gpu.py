import os
import numpy as np
import tensorflow as tf
import argparse
from tensorflow import keras
from tensorflow.python.ops import array_ops
from time import time
#%%
def searchDirectory(ret,directory,filename):
    for t in os.listdir(directory):
        if os.path.isdir(os.path.join(directory,t)):
            searchDirectory(ret,os.path.join(directory,t),filename)
        else:
            if filename in t :
                ret.append(os.path.join(directory,t))
    return
#%%
def dataset_func(filename):
    tmp = np.load(filename.numpy())
    data = tmp[:, :40]
    label = tmp[:, 40] +0.2
        
    return data, label
#%%
def process_path(file_path):
    [data, label,] = tf.py_function(dataset_func, [file_path], (tf.float32, tf.float32))
    return data, label


#%%
def cross_entropy_loss(labels, preds):
    labels = tf.reshape(labels, [-1])
    mask = tf.cast(tf.sign(labels), tf.float32)
    labels = tf.cast(labels, tf.int64)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=preds)
    loss *= mask
    loss = tf.reduce_sum(loss)
    loss /= tf.reduce_sum(mask)
    
    return loss

#%%
def train_accuracy(labels, preds):
    labels = tf.reshape(labels, [-1])
    mask = tf.cast(tf.sign(labels), tf.float32) # maksing labels which is 0. tf.mask return 0 when the input is 0
    #tf.print(mask,output_stream=sys.stderr)
    labels = tf.cast(labels, tf.int64)
    acc = tf.cast(tf.equal(labels, tf.argmax(preds, 1)), tf.float32) 
    acc *= mask
    acc = tf.reduce_sum(acc) #positive true
    acc /= tf.reduce_sum(mask) 
    return acc

#%%
def test_accuracy(labels, preds):
    labels = tf.reshape(labels, [-1])
    mask = tf.cast(tf.sign(labels), tf.float32)
    labels = tf.cast(labels, tf.int64)
    preds = tf.argmax(preds, 1)
    intersliencemask = tf.cast(preds==1, tf.float32)*-1+1
    mask = mask * intersliencemask
    
    preds *= -1
    preds = tf.sign(preds) + 1
    
    acc = tf.cast(tf.equal(labels, preds), tf.float32)
    acc *= mask
    acc = tf.reduce_sum(acc)
    acc /= tf.reduce_sum(mask)
    return acc

#%%
def decay(epoch):
    if epoch < 100:
        return 0.01
    elif epoch >= 100 and epoch < 300:
        return 0.005
    else:
        return 1.0/float(epoch)

#%%
class LSTM(tf.keras.Model):
    def __init__(self, num_layers = 5, units=16):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.units = units
        self.lstm_cells = [tf.keras.layers.LSTM(units=self.units, return_sequences = True) for i in range(num_layers)]
        self.dnn1 = tf.keras.layers.Dense(self.units, activation = 'relu')
        self.dnn2 = tf.keras.layers.Dense(2)
        self.drop = tf.keras.layers.Dropout(0.5)

    def call(self, x):
        for i in range(self.num_layers):
            x = self.lstm_cells[i](x)
        x = tf.reshape(x, [-1, self.units])
        x = self.dnn1(x)
        x = self.drop(x)
        x = self.dnn2(x)
        return x

#%%
class GRU(tf.keras.Model):
    def __init__(self, num_layers = 1, units = 32):
        super(GRU, self).__init__()
        self.num_layers = num_layers
        self.units = units
        self.gru_cells = [tf.keras.layers.GRU(units=self.units, return_sequences = True) for i in range(num_layers)]
        self.dnn1 = tf.keras.layers.Dense(self.units, activation = 'relu')
        self.dnn2 = tf.keras.layers.Dense(2)
        self.drop = tf.keras.layers.Dropout(0.5)

    def call(self, x):
        for i in range(self.num_layers):
            x = self.gru_cells[i](x)
        x = tf.reshape(x, [-1, self.units])
        x = self.dnn1(x)
        x = self.drop(x)
        x = self.dnn2(x)
        return x


#%%
class Mobile(tf.keras.Model):
    def __init__(self, grid_lstm_cell):
        
        super(GDLLDNN, self).__init__()
        
        self.grid_lstm_cell = grid_lstm_cell
        self.dnn1 = tf.keras.layers.Dense(64, activation = 'relu')
        self.lstm_cells = [tf.keras.layers.LSTM(units=64, return_sequences = True) for i in range(2)]
        self.dnn2 = tf.keras.layers.Dense(64, activation = 'relu')
        self.dnn3 = tf.keras.layers.Dense(2)
        self.drop = tf.keras.layers.Dropout(0.3)


    def call(self, inputs):
        
        seq_size = array_ops.shape(inputs)[1]
        
        x = keras.layers.RNN(self.grid_lstm_cell, return_sequences = True)(inputs)
        x = tf.reshape(x, [-1, 408])
        x = self.dnn1(x)
        x = self.drop(x)
        x = tf.reshape(x, [-1, seq_size, 64])
        
        for i in range(2):
            x = self.lstm_cells[i](x)
            
        x = tf.reshape(x, [-1, 64])
        x = self.dnn2(x)
'''
#%%
def main(config):

    mirrored_strategy = tf.distribute.MirroredStrategy()

    datalist = []
    searchDirectory(datalist, config.train_data_path, '.npy')
    
    rng = np.random.RandomState(123)
    rng.shuffle(datalist)
    rng.shuffle(datalist)    
    rng.shuffle(datalist)

    print(len(datalist))

    with mirrored_strategy.scope():
        dataset = tf.data.Dataset.from_tensor_slices((datalist))
        dataset = dataset.shuffle(len(datalist))
        dataset = dataset.repeat()
        dataset = dataset.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.padded_batch(int(config.batch_size), padded_shapes=((None, 40), (None, )))
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    
    testdatalist = []
    #searchDirectory(testdatalist,'./datas/val/luxone/vc_normal','.npy')
    #searchDirectory(testdatalist,'./datas/val/phone/new_tc/50cm','.npy')
    #searchDirectory(testdatalist,'./datas/val/phone/new_tc/1m','.npy')
    #searchDirectory(testdatalist,'./datas/val/phone/new_tc/oov','.npy')
    #searchDirectory(testdatalist,'./datas/val/tablet/new_tc','.npy')
    searchDirectory(testdatalist,'./datas/val/watch/clean','.npy')
    searchDirectory(testdatalist,'./datas/val/watch/noise','.npy')
    searchDirectory(testdatalist,'./datas/val/watch/oov','.npy')
    print(len(testdatalist))
    rng = np.random.RandomState(123)
    rng.shuffle(testdatalist)

    remain = len(testdatalist)%8
    size = len(testdatalist) - remain

    with mirrored_strategy.scope():
        testdataset = tf.data.Dataset.from_tensor_slices((testdatalist))
        testdataset = testdataset.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        testdataset = testdataset.padded_batch(len(testdatalist),  padded_shapes=((None, 40), (None, )))
        testdataset = testdataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True),
        tf.keras.callbacks.LearningRateScheduler(decay),
        #tf.keras.callbacks.TensorBoard(log_dir="logs/{}".format(time()))
        #PrintLR
    ]

    with mirrored_strategy.scope():
         
        model = GRU(num_layers = 1,units=32) 
        #model = LSTM(num_layers = 1) 

        #model = tf.keras.experimental.load_from_saved_model(checkpoint_dir)
        
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                      loss=cross_entropy_loss,
                      metrics=[train_accuracy])

    #model.summary()
    #latest = tf.train.latest_checkpoint(checkpoint_dir, )
    #latest = "./training_checkpoints/ckpt_{0}".format(100)
    #model.load_weights(latest)
    
    model.fit(dataset, epochs=2000, steps_per_epoch = 30, callbacks=callbacks,validation_data=testdataset )
    
    
    model.summary()
    
#%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path',  default='./datas/train/rir')
    parser.add_argument('--batch_size',       default= 1000)
    parser.add_argument('--learning_rate',    default= 0.001)
    parser.add_argument('--test_data_path',   default='./data/val/100_7300/vc_normal')
    config = parser.parse_args()
    main(config)
