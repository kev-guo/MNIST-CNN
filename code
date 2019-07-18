import numpy as np
import keras as ker

from keras.datasets import mnist
import matplotlib.pyplot as plt
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

#for reproducible results
seed = 1
np.random.seed(seed)

#prints out sample image from database
plt.subplot(221)
plt.imshow(X_train[0], cmap = plt.get_cmap('gray'))
plt.show()

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

#use theano for tensor shape, try "image_dim_ordering" for updated API
K.set_image_dim_ordering('th')

#reshape to [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1 , 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

#change pixel value range to 0 to 1
X_train /= 225
X_test /= 225

#just in case the dataset is categorical, convert to numerical
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)
#number of classes to use for dense layer
num_classes = Y_test.shape[1]
num_classes = 10


"""CODE TO TRAIN NETWORK"""
#define model and create seq model
def model():
    mod = Sequential()
    #30 feature maps with a size 5x5 (for 1st layer, add input_shape (pixels, width, height))
    #relu works better than softmax for this CNN
    mod.add(Conv2D(30, (5, 5), 
            input_shape = (1, 28, 28), 
            activation = 'relu'))
    #downsample
    mod.add(MaxPooling2D(pool_size = (2,2)))            
    #2nd convolutional layer
    mod.add(Conv2D(15, (3, 3), 
            activation = 'relu'))
    #downsample
    mod.add(MaxPooling2D(pool_size = (2,2)))
    #mod.add(Dropout(0.2)) <- reduce overfitting if needed
    #convert to 1D vector by flattening
    mod.add(Flatten())
    
    #dense layers with 128 hidden units
    mod.add(Dense(128, activation = 'relu'))
    mod.add(Dense(128, activation = 'relu'))
    #last layer has 10 nodes, for results 0-9
    mod.add(Dense(num_classes, activation = "softmax"))
    
    #compile to configure model for training
    mod.compile(optimizer = 'adam',
                loss = 'categorical_crossentropy',
                metrics = ['accuracy'],
                loss_weights = None,
                sample_weight_mode = None,
                weighted_metrics = None,
                target_tensors = None)
    return mod
    
#model is configured to fit 10 epochs, batch size = 200 per
mod = model()
mod.fit(X_train, Y_train, validation_data = (X_test, Y_test), epochs = 10, batch_size = 200)


"""CODE TO EVALUATE NETWORKd"""
score = mod.evaluate(X_test, Y_test, verbose = 0)
print("Percent error: %.2f%%" % (100 - score[1] * 100))
