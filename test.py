import data_improvements as di
import pandas as pd

data = pd.read_csv('data/train.csv').drop('target', axis=1)
#train_data = pd.read_csv('data/train.csv').drop('id', axis=1).drop('target', axis=1).dropna(axis=1)

print(data)

for i in ['RES_ANNIMP', 'id']:
    data = data.drop(i, axis=1)

for i in ['CTR_CATEGO_X']:
    print(1, i)
    data = di.classify(data, i)

for i in di.nan_columns(data):
    print(2, i)
    data = di.denanization(data, i)

for i in data.columns:
    print(3, i)
    data = di.normalize(data, i)

data = di.compare(data)

data.to_csv('data/train_converted.csv')