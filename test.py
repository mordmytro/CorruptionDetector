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

train_data = pd.read_csv('data/train.csv', index_col=None).dropna(axis=1)
#test_data = pd.read_csv('data/test.csv', index_col=None)

targets = train_data['target']
train_data = train_data.drop('target', axis=1).drop('id', axis=1)#.drop('CTR_CATEGO_X', axis=1)

#%% Edit CTR_CATEGO_X column
def remake_column_with_symbols(data, col):
    if col in data.columns:
        tmp_list = []
        for i in data[col]:
            if i not in tmp_list:
                #print(i)
                tmp_list.append(i)
        
        tmp_list.sort()
        print(tmp_list)
        
        dc = {}
        
        for name in tmp_list:
            dc[name] = []
        
        for i in data[col]:
            for j in dc:
                if i == j:
                    dc[j].append(1)
                else:
                    dc[j].append(0)
        
        new_list = []
        for i in dc:
            new_list.append(dc[i])
        
        for i in range(len(tmp_list)):
            tmp_list[i] = col + str(tmp_list[i])
            #print(tmp_list[i])
        
        #print(new_list)
        
        df = pd.DataFrame(np.array(new_list).T.tolist(), columns=tmp_list)
        
        num = data.columns.get_loc(col)
        data = data.drop(col, axis=1)
        
        for i in df:
            data.insert(num, i, df[i].values)
    return data
#%%

train_data = remake_column_with_symbols(train_data, 'CTR_CATEGO_X')

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