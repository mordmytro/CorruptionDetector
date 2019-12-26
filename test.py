import data_improvements as di
import pandas as pd

data = pd.read_csv('data/train.csv').drop('target', axis=1)
#train_data = pd.read_csv('data/train.csv').drop('id', axis=1).drop('target', axis=1).dropna(axis=1)

to_classify = ['CTR_CATEGO_X', 'CTR_CESSAT', 'CTR_OBLDIR', 'CTR_OBLACP', 
               'CTR_OBLRES', 'CTR_OBLFOP', 'CTR_OBLTFP', 'CTR_OBLDCO', 
               'CTR_OBLTVA', 'CTR_OFODEC', 'CTR_OFODEP', 'CTR_OFODET', 
               'CTR_OBLAUT', 'CTR_OBLASS', 'CTR_ODTIMB', 'CTR_OBLTCL',
               'CTR_OBLTHO', 'CTR_OBLDLI', 'CTR_OBLTVI', 'CTR_RATISS',
               'EXE_EXERCI', 'TVA_MOIDEB', 'TVA_MOIFIN']

unkown = ['BCT_CODBUR', 'FJU_CODFJU']

to_drop = ['RES_ANNIMP', 'id']

print(data)

for i in to_drop:
    data = data.drop(i, axis=1)

for i in to_classify:
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