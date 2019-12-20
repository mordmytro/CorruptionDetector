import keras
import pandas as pd
import numpy as np

train_data = pd.read_csv('data/train.csv', index_col=None)
test_data = pd.read_csv('data/test.csv', index_col=None)