# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 15:28:07 2016

@author: gazkune
"""

import numpy as np
from numpy.random import choice

from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences


X = np.zeros([1200, 20])
#N_train = 1000
one_indexes = choice(a=1200, size=1200 / 2, replace=False)
X[one_indexes, 0] = 1  # very long term memory.

print 'X shape:', X.shape
print X

X = X.reshape(X.shape[0], X.shape[1], 1)
print 'X reshape:', X.shape

y = X[:, 0]

print 'y shape:', y.shape
print y

total_examples = len(X)
test_per = 0.2
limit = int(test_per * total_examples)
X_train = X[limit:]
X_test = X[:limit]
y_train = y[limit:]
y_test = y[:limit]

print 'X_train shape:', X_train.shape
print X_train

print 'Total examples:', total_examples
print 'Train examples:', len(X_train), len(y_train) 
print 'Test examples:', len(X_test), len(y_test)

"""
sample = np.expand_dims(np.expand_dims(X_train[0][0], axis=1), axis=1)
print 'sample shape:', sample.shape
print sample

other = X_train[0][0]
print 'other shape:', other.shape
print other
"""


batch_size = 10

print('Build STATELESS model...')
model = Sequential()
model.add(LSTM(10, input_shape=(X.shape[1], 1), return_sequences=False))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print 'Model built'
print(model.summary())

print "Training..."
epochs = 15
history = model.fit(X, y, batch_size=batch_size, nb_epoch=epochs, validation_data=(X_test, y_test), shuffle=True)
