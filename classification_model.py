from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow и tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Вспомогательные библиотеки
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import json

import data_improvements as di

train_data = pd.read_csv('data/train_converted.csv').drop('Unnamed: 0', axis=1).dropna(axis=1)
#train_data = pd.read_csv('data/train.csv').drop('id', axis=1).drop('target', axis=1).dropna(axis=1)

#train_data.to_csv('data/train_converted.csv')
#test_data = pd.read_csv('data/test.csv', index_col=None)

targets = list(pd.read_csv('data/train.csv', index_col=None)['target'])

#train_data = pd.read_csv('data/train_converted.csv').drop('Unnamed: 0', axis=1)
#test_data = pd.read_csv('data/test_converted.csv').drop('Unnamed: 0', axis=1)
#targets = np.array([0]*len(train_data) + [1]*len(test_data))


for i in range(len(targets)):
    if targets[i] != 0:
        targets[i] = [1, 0]
    else:
        targets[i] = [0, 1]

targets = np.array(targets)


def build_model():
    model = Sequential()
    
    model.add(Dense(128, activation='relu', input_shape=(len(list(train_data.columns)), ), kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(Dense(2, activation='softmax'))

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

def build_classification_model():

    early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=25)

    inputs = train_data
    outputs = targets
    model = build_model()
    
    print(inputs, outputs)

    history = model.fit(
        inputs,
        outputs,
        epochs=30,
        validation_split=0.2,
        #callbacks=[early_stop]
    )

    #plot_history(history)

    
    def test_coef(coef = 0.5):
        test_predictions = model.predict(train_data.iloc[int(len(train_data)*0.7):])
        test_labels = targets[int(len(train_data)*0.7):]

        good_predicted_good = 0
        good_predicted_bad = 0
        bad_predicted_good = 0
        bad_predicted_bad = 0

        for i, j in zip(test_predictions, test_labels):
            label = 1 if j[1] == 1 else 0 # 0 means good, 1 means corruption
            prediction = 0 if i[0] > coef else 1
            if label == prediction == 0:
                good_predicted_good += 1
            elif label == prediction == 1:
                bad_predicted_bad += 1
            elif label == 1 and prediction == 0:
                bad_predicted_good += 1
            elif label == 0 and prediction == 1:
                good_predicted_bad += 1

        total_predictions = good_predicted_good + bad_predicted_bad + good_predicted_bad + bad_predicted_good
        total_correct = good_predicted_good + bad_predicted_bad
        total_incorrect = total_predictions - total_correct
        total_good = good_predicted_good + good_predicted_bad
        total_bad = bad_predicted_good + bad_predicted_bad

        print(f'------------Results {coef}-------------')
        print(f'Correct predictions: {round(100 * total_correct/total_predictions if total_predictions != 0 else 0, 5)}% ({total_correct}/{total_predictions})')
        print(f'Incorrect predictions: {round(100 * total_incorrect/total_predictions if total_predictions != 0 else 0, 5)}% ({total_incorrect}/{total_predictions})')
        print(f'Not corrupted predicted correct: {round(100 * good_predicted_good/total_good if total_predictions != 0 else 0, 5)}% ({good_predicted_good}/{total_good})')
        print(f'Not corrupted predicted incorrect: {round(100 * good_predicted_bad/total_good if total_predictions != 0 else 0, 5)}% ({good_predicted_bad}/{total_good})')
        print(f'Corrupted predicted correct: {round(100 * bad_predicted_bad/total_bad if total_predictions != 0 else 0, 5)}% ({bad_predicted_bad}/{total_bad})')
        print(f'Corrupted predicted incorrect: {round(100 * bad_predicted_good/total_bad if total_predictions != 0 else 0, 5)}% ({bad_predicted_good}/{total_bad})')
        print('--------------------------------------')

    #for i in np.arange(0, 1, 0.05):
    #    test_coef(i)

    return model



if __name__ == "__main__":
    build_classification_model()