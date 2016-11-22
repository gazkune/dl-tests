# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 11:33:04 2016

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

sample = np.expand_dims(np.expand_dims(X_train[0][0], axis=1), axis=1)
print 'sample shape:', sample.shape
print sample

other = X_train[0][0]
print 'other shape:', other.shape
print other



batch_size = 1

print('Build STATEFUL model...')
model = Sequential()
model.add(LSTM(10, batch_input_shape=(1, 1, 1), return_sequences=False, stateful=True))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

max_len = X.shape[1]
print('Train...')
for epoch in range(15):
    print '***************'
    print 'Epoch', epoch, '/ 15'
    mean_tr_acc = []
    mean_tr_loss = []
    for i in range(len(X_train)):
        y_true = y_train[i]
        for j in range(max_len):
            tr_loss, tr_acc = model.train_on_batch(np.expand_dims(np.expand_dims(X_train[i][j], axis=1), axis=1),
                                                   np.array([y_true]))
            mean_tr_acc.append(tr_acc)
            mean_tr_loss.append(tr_loss)
        model.reset_states()

    print('accuracy training = {}'.format(np.mean(mean_tr_acc)))
    print('loss training = {}'.format(np.mean(mean_tr_loss)))
    print('___________________________________')

    mean_te_acc = []
    mean_te_loss = []
    for i in range(len(X_test)):
        for j in range(max_len):
            te_loss, te_acc = model.test_on_batch(np.expand_dims(np.expand_dims(X_test[i][j], axis=1), axis=1),
                                                  y_test[i])
            mean_te_acc.append(te_acc)
            mean_te_loss.append(te_loss)
        model.reset_states()

        for j in range(max_len):
            y_pred = model.predict_on_batch(np.expand_dims(np.expand_dims(X_test[i][j], axis=1), axis=1))
        model.reset_states()

    print('accuracy testing = {}'.format(np.mean(mean_te_acc)))
    print('loss testing = {}'.format(np.mean(mean_te_loss)))
    print('___________________________________')
