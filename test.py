from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow и tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Вспомогательные библиотеки
import numpy as np
import matplotlib.pyplot as plt

import keras
import pandas as pd
import numpy as np
import json

train_data = pd.read_csv('data/train.csv', index_col=None)
#test_data = pd.read_csv('data/test.csv', index_col=None)

targets = train_data['target']
train_data = train_data.drop('target', axis=1).drop('id', axis=1).drop('CTR_CATEGO_X', axis=1)

def build_model():
    model = Sequential()
    
    model.add(Dense(16, activation='relu', input_shape=(1, )))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(
        loss='mse',
        optimizer=optimizer,
        metrics=['mae', 'mse']
    )
      
    return model

def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mae'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'],
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

early_stop = keras.callbacks.EarlyStopping(monitor='val_mse', patience=10)

results = {}

for column in train_data.columns:
    inputs = train_data[column]
    outputs = targets
    model = build_model()
    
    history = model.fit(
        inputs,
        outputs,
        epochs=200,
        validation_split = 0.2,
        callbacks=[early_stop]
    )
    
    hist = pd.DataFrame(history.history)
    
    results[column] = min(hist['val_mae'])
    
    plot_history(history)
    
    print(results)
        
    with open('data.json', 'w') as fp:
        json.dump(results, fp)