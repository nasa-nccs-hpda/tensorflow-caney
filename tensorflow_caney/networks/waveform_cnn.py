#import tensorflow as tf
#from keras.layers import *
#from keras.optimizers import *
#from keras.models import *
#from keras.callbacks import *
#from keras.regularizers import *
#import keras.backend as K

import logging
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv1D, MaxPooling2D, Dropout
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import concatenate, Input, UpSampling2D
from tensorflow.keras.layers import BatchNormalization, Add, Reshape
from tensorflow.keras.layers import Dense, Multiply, Flatten
from tensorflow.keras.regularizers import l2

# source from bpowell
 
def resblockconv(channels, inputlayer):
    x = Conv1D(channels, kernel_size=4, strides=2, padding='same')(inputlayer)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=.1)(x)
    x = Conv1D(channels, kernel_size=4, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=.1)(x)
    x = Conv1D(channels, kernel_size=4, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    return x

def resblockconvsc(channels, inputlayer):
    x = Conv1D(channels, kernel_size=4, strides=2, padding='same')(inputlayer)
    x = BatchNormalization()(x)
    return x

def resblockid(channels, inputlayer):
    x = Conv1D(channels, kernel_size=8, strides=1, padding='same')(inputlayer)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=.1)(x)
    x = Conv1D(channels, kernel_size=8, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=.1)(x)
    x = Conv1D(channels, kernel_size=8, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    return x
 
def fullresblock(channels, inputlayer):
    xr = resblockconv(channels, inputlayer)
    xrsc = resblockconvsc(channels, inputlayer)
    xadd = Add()([xr, xrsc])
    xact = LeakyReLU(alpha=.1)(xadd)
    xid = resblockid(channels, xact)
    xadd = Add()([xact, xid])
    xact = LeakyReLU(alpha=.1)(xadd)
    return xact

def waverform_cnn(lcsize: int = 100):

    # defining Input layer
    input_layer = Input(shape=(lcsize, 1))

    x = Reshape((lcsize,))(input_layer)
    attn = Dense(lcsize, activation='softmax')(x)
    mult = Multiply()([attn, x])
    x = Add()([mult, x])
    x = Reshape((lcsize, 1))(x)
    
    x = fullresblock(16,x)
    # x = fullresblock(16,input_layer)
    x=Dropout(.5, noise_shape=(1, 16))(x)
    
    x = fullresblock(32, x)
    x = Dropout(.5, noise_shape=(1, 32))(x)
    
    x = fullresblock(48, x)
    x = Dropout(.5, noise_shape=(1, 48))(x)
    
    x = fullresblock(54, x)
    x = Dropout(.5, noise_shape=(1, 54))(x)
    
    x = fullresblock(70, x)
    x = Dropout(.5, noise_shape=(1, 70))(x)
    
    x = fullresblock(86, x)
    x = Dropout(.5, noise_shape=(1, 86))(x)
    
    x = fullresblock(102, x)
    x = Dropout(.5, noise_shape=(1, 102))(x)
    
    x = fullresblock(118, x)
    x = Dropout(.5, noise_shape=(1, 118))(x)
    
    x = fullresblock(134, x)
    x = Dropout(.5, noise_shape=(1, 134))(x)
    
    x = fullresblock(150, x)
    x = Flatten()(x)
    x = Dropout(.5)(x)
    x = Dense(1024)(x)
    x = LeakyReLU(alpha=.1)(x)
    x = Dropout(.5)(x)
    output_layer = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input_layer, outputs=output_layer, name="1D-CNN")

    return model


# -------------------------------------------------------------------------------
# module unet Unit Tests
# -------------------------------------------------------------------------------

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    # Can add different images sizes - for now: (256,256,6)
    waverform_model = waverform_cnn(100)
    waverform_model.summary()
