import data_improvements as di
import pandas as pd

train_data = pd.read_csv('data/train.csv').drop('target', axis=1)
test_data = pd.read_csv('data/test.csv')#.drop('target', axis=1)
#train_data = pd.read_csv('data/train.csv').drop('id', axis=1).drop('target', axis=1).dropna(axis=1)

data, ind = di.append_data(train_data, test_data)

to_classify = ['CTR_CATEGO_X', 'CTR_CESSAT', 'CTR_OBLDIR', 'CTR_OBLACP', 
               'CTR_OBLRES', 'CTR_OBLFOP', 'CTR_OBLTFP', 'CTR_OBLDCO', 
               'CTR_OBLTVA', 'CTR_OFODEC', 'CTR_OFODEP', 'CTR_OFODET', 
               'CTR_OBLAUT', 'CTR_OBLASS', 'CTR_ODTIMB', 'CTR_OBLTCL',
               'CTR_OBLTHO', 'CTR_OBLDLI', 'CTR_OBLTVI', 'CTR_RATISS',
               'EXE_EXERCI', 'TVA_MOIDEB', 'TVA_MOIFIN']

to_zero_or_one = []
"""'TVA_CHAFF6', 'TVA_CHAFF7', 'TVA_CHAF10', 'TVA_CHAF12',
'TVA_CAF125', 'TVA_CHAF15', 'TVA_CHAF18', 'TVA_CHAF22', 
'TVA_CHAF29', 'TVA_CHAF36', 'TVA_TOTDUE', 'TVA_CRDINI', 
'TVA_BASIMB', 'TVA_DEDIMB', 'TVA_BASEQL', 'TVA_DEDEQL',
'TVA_BASEQI', 'TVA_DEDEQI', 'TVA_BASAUL', 'TVA_DEDAUL', 
'TVA_BASAUI', 'TVA_DEDAUI', 'TVA_BASRSM', 'TVA_RSNRES', 
'TVA_TRSPOR', 'TVA_DEDREG', 'TVA_RESTIT', 'TVA_MNTPAY',
'TVA_MOIFIN', 'TVA_CRDFIN', 'TVA_ACHSUS', 'TVA_ACHEXO',
'']"""

for i in data.columns:
    if i.startswith('TVA_') or i.startswith('AX'):
        to_zero_or_one.append(i)

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

for i in to_zero_or_one:
    print(3, i)
    data = di.one_or_null(data, i)

for i in data.columns:
    print(4, i)
    data = di.normalize(data, i)

data = di.compare(data)

train_data_converted = data[:ind]
test_data_converted = data[ind:]

print(train_data_converted)
print(test_data_converted)

train_data_converted.to_csv('data/train_converted.csv')
test_data_converted.to_csv('data/test_converted.csv')