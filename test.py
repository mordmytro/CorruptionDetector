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

train_data = pd.read_csv('data/train.csv', index_col=None)#.dropna(axis=1)
#test_data = pd.read_csv('data/test.csv', index_col=None)

targets = train_data['target']
train_data = train_data.drop('target', axis=1).drop('id', axis=1)#.drop('CTR_CATEGO_X', axis=1)

#%% Classify column

def classify(start_data, col):
    data = start_data.copy()
    if col in data.columns:
        tmp_list = []
        for i in data[col]:
            if i not in tmp_list:
                tmp_list.append(i)
        
        tmp_list.sort()        
        dc = {}
        
        for name in tmp_list:
            dc[name] = []
        
        for i in data[col]:
            for j in dc:
                dc[j].append(1) if i == j else dc[j].append(0)

        new_list = []
        for i in dc:
            new_list.append(dc[i])
        
        for i in range(len(tmp_list)):
            tmp_list[i] = col + str(tmp_list[i])
        
        df = pd.DataFrame(np.array(new_list).T.tolist(), columns=tmp_list)
        
        num = data.columns.get_loc(col)
        data = data.drop(col, axis=1)
        
        for i in df:
            data.insert(num, i, df[i].values)
    return data

#%% Remove Nans and make a new boolean is_Nan column
    
def denanization(start_data, col):
    data = start_data.copy()
    tmp_list = []
    num = data.columns.get_loc(col)
    for i in range(len(data[col])):
        if np.isnan(data[col].iloc[i]):
            tmp_list.append(1)
            data[col].iloc[i] = 0
        else:
            tmp_list.append(0)
    data.insert(num + 1, col + '_is_NaN', tmp_list)
    #print(data)
    return data

#%% Making all list values in range from 0 to 1

def normalize(start_data, col):
    data = start_data.copy()
    max_val = max(data[col].values)
    min_val = min(data[col].values)
    
    for i in range(len(data[col])):
        data[col].iloc[i] = (data[col].iloc[i] - min_val) / (max_val - min_val)
    
    return data

#%%

train_data = classify(train_data, 'CTR_CATEGO_X')

def build_model():
    model = Sequential()
    
    model.add(Dense(128, activation='relu', input_shape=(len(train_data.columns), )))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
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

early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=40)

inputs = train_data
outputs = targets
model = build_model()

history = model.fit(
    inputs,
    outputs,
    epochs=200,
    validation_split = 0.1,
    callbacks=[early_stop]
)

plot_history(history)

test_predictions = model.predict(train_data.iloc[int(len(train_data)*0.9):]).flatten()
test_labels = targets.iloc[int(len(train_data)*0.9):]

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
plt.show()