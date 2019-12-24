from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow и tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Вспомогательные библиотеки
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import json

import data_improvements as di

train_data = pd.read_csv('data/train_converted.csv').drop('Unnamed: 0', axis=1).dropna(axis=0)
#train_data = pd.read_csv('data/train.csv').drop('id', axis=1).drop('target', axis=1).dropna(axis=1)

#print(train_data.dtypes)

#for column in di.not_number_columns(train_data):
#    train_data = di.classify(train_data, column)
 
for column in train_data.columns:
    print(column)
    train_data = di.normalize(train_data, column)
    
train_data.to_csv('data/train_converted.csv')

#test_data = pd.read_csv('data/test.csv', index_col=None)

targets = np.array(pd.read_csv('data/train.csv', index_col=None)['target'])

def build_model():
    model = Sequential()
    
    model.add(Dense(128, activation='relu', input_shape=(len(list(train_data.columns)), )))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1))

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(
        loss='mse',
        optimizer = 'adam',
        metrics=['mae', 'mse']
    )
      
    return model

def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    
    print(hist)
    
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
             label = 'Val Error')
    plt.legend()
    
    '''
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mse'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'],
             label = 'Val Error')
    plt.legend()
    '''
    plt.show()

if __name__ == '__main__':

    early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=20)

    inputs = train_data
    outputs = targets
    model = build_model()
    
    print(inputs, outputs)

    history = model.fit(
        inputs,
        outputs,
        epochs=200,
        validation_split = 0.1,
        callbacks=[early_stop]
    )

    plot_history(history)

    test_predictions = model.predict(train_data.iloc[int(len(train_data)*0.9):]).flatten()
    test_labels = targets[int(len(train_data)*0.9):]

    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True Values [MPG]')
    plt.ylabel('Predictions [MPG]')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0,plt.xlim()[1]])
    plt.ylim([0,plt.ylim()[1]])
    _ = plt.plot([-10000, 10000], [-10000, 10000])
    plt.show()