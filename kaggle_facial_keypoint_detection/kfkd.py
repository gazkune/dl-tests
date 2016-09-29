# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 12:49:10 2016

@author: gazkune
"""

import os, sys

import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

# Keras imports
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, MaxPooling2D, Flatten, Activation
from keras.optimizers import SGD

FTRAIN = './data/training.csv'
FTEST = './data/test.csv'

"""
Function to load the training and testing datasets
"""
def load(test=False, cols=None):
    """Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    Pass a list of *cols* if you're only interested in a subset of the
    target columns.
    """
    fname = FTEST if test else FTRAIN
    df = read_csv(os.path.expanduser(fname))  # load pandas dataframe

    # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    if cols:  # get a subset of columns
        df = df[list(cols) + ['Image']]

    print(df.count())  # prints the number of values for each column
    df = df.dropna()  # drop all rows that have missing values in them

    X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)

    if not test:  # only FTRAIN has any target columns
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48  # scale target coordinates to [-1, 1]
        X, y = shuffle(X, y, random_state=42)  # shuffle train data
        y = y.astype(np.float32)
    else:
        y = None

    return X, y

"""
Function to show the results of the net prediction in testing images
"""

def plot_sample(x, y, axis):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)
    
"""
Function to load data and format in order to be used with convolutional
neural networks
"""

def load2d(test=False, cols=None):
    X, y = load(test=test)
    X = X.reshape(-1, 1, 96, 96)
    return X, y
    
"""
Function to plot accurary and loss during training
"""

def plot_training_info(metrics, save, history):
    # summarize history for accuracy
    if 'accuracy' in metrics:
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        if save == True:
            plt.savefig('accuracy.png')
        else:
            plt.show()

    # summarize history for loss
    if 'loss' in metrics:
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.ylim(1e-3, 1e-2)
        plt.yscale("log")
        plt.legend(['train', 'test'], loc='upper left')
        if save == True:
            plt.savefig('loss.png')
        else:
            plt.show()
    

def main(argv):
    """
    # Load data as a 1D vector
    X, y = load()
    print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(
        X.shape, X.min(), X.max()))
    print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
        y.shape, y.min(), y.max()))
    
    # A fully-connected feed forward neural network    
    net1 = Sequential()
    net1.add(Dense(100, input_dim=9216, activation='relu')) # input+hidden layer?
    net1.add(Dense(30)) # output layer

    net1.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True), metrics=['accuracy'])
    history = net1.fit(X, y, validation_split=0.2, nb_epoch=30, batch_size=128)



    # test net1 with the testing dataset
    X, _ = load(test=True)
    y_pred = net1.predict(X)

    fig = plt.figure(figsize=(6, 6))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(16):
        ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
        plot_sample(X[i], y_pred[i], ax)

    plt.show()
    """

    """
    Model, train and test a Convolutional Neural Network
    """
    # load data as 2D matrix for the convolution layer
    X, y = load2d()  

    net2 = Sequential()
    # Input and conv1 + maxpool1
    net2.add(Convolution2D(32, 3, 3, input_shape=(1, 96, 96), border_mode='valid'))
    net2.add(MaxPooling2D(pool_size=(2, 2)))
    # conv2 + pool2
    net2.add(Convolution2D(64, 2, 2, border_mode='valid'))
    net2.add(MaxPooling2D(pool_size=(2, 2)))
    # conv3 + pool3
    net2.add(Convolution2D(128, 2, 2, border_mode='valid'))
    net2.add(MaxPooling2D(pool_size=(2, 2)))
    net2.add(Flatten())
    # hidden4
    net2.add(Dense(500))
    # hidden5
    net2.add(Dense(500))
    # Output layer
    net2.add(Dense(30))

    net2.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True), metrics=['accuracy'])
    print(net2.summary())
    
    history = net2.fit(X, y, validation_split=0.2, nb_epoch=1, batch_size=128)
    
    # serialize model to JSON
    model_json = net2.to_json()
    with open("net2.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    net2.save_weights("net2.h5")
    print("Saved model to disk")
    
    plot_training_info('loss', True, history)

if __name__ == "__main__":
   main(sys.argv)