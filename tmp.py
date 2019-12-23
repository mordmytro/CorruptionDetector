import pandas as pd
import numpy as np

train_data = pd.read_csv('data/train.csv', index_col=False)
'''
tmp_list = []
for i in train_data['CTR_CATEGO_X']:
    if i not in tmp_list:
        print(i)
        tmp_list.append(i)

print(tmp_list)

dc = {}

for name in tmp_list:
    dc[name] = []

for i in train_data['CTR_CATEGO_X']:
    for j in dc:
        if i == j:
            dc[j].append(1)
        else:
            dc[j].append(0)

new_list = []
for i in dc:
    new_list.append(dc[i])

for i in range(len(tmp_list)):
    tmp_list[i] = 'CTR_CATEGO_X' + tmp_list[i]
    print(tmp_list[i])

#print(new_list)

df = pd.DataFrame(np.array(new_list).T.tolist(), columns=tmp_list)

num = train_data.columns.get_loc("CTR_CATEGO_X")
train_data = train_data.drop('CTR_CATEGO_X', axis=1)

for i in df:
    train_data.insert(num, i, df[i].values)
'''
def denanization(data, col):
    tmp_list = []
    num = data.columns.get_loc(col)
    for i in range(len(data[col])):
        if np.isnan(data[col].iloc[i]):
            tmp_list.append(1)
            data[col].iloc[i] = 0
        else:
            tmp_list.append(0)
    data.insert(num + 1, col + '_is_NaN', tmp_list)
    
    
#denanization(train_data, 'FAC_MNTTVA_C')

'''
def normalizing(data, col):
    max_val = max(data[col].values)
    min_val = min(data[col].values)
    print(min_val, max_val)

    print(data[col])
    
    for i in range(len(data[col])):
        data[col].iloc[i] = (data[col].iloc[i] - min_val) / (max_val - min_val)
    
    print(data[col])
    
normalizing(train_data, 'BCT_CODBUR')
'''

def is_not_number(data):
    tmp_list = []
    for i in data:
        if data[i].dtypes != float and data[i].dtypes != int:
            tmp_list.append(i)
    return tmp_list

def is_not_nan(data):
    tmp_list = []
    asa = []
    for i in data:
        if False in data[i].notna().values:
            tmp_list.append(i)
        else:
            asa.append(i)
    return tmp_list, asa

print(is_not_nan(train_data))