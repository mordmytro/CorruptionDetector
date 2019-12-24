import pandas as pd
import numpy as np

def classify(start_data, col):
    '''
    Splits classification column into separate boolean columns
    '''
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

def denanization(start_data, col):
    '''
    Splits column into is NaN boolean column and value column
    '''
    data = start_data.copy()
    tmp_list = []
    num = data.columns.get_loc(col)
    print(num)
    for i in range(len(data[col])):
        if np.isnan(data[col].iloc[i]):
            tmp_list.append(1)
            data[col].iloc[i] = 0
        else:
            tmp_list.append(0)
    data.insert(num + 1, col + '_is_NaN', tmp_list)
    return data

def normalize(start_data, col):
    '''
    Normalizes data in the column
    '''
    data = start_data.copy()
    max_val = max(data[col].values)
    min_val = min(data[col].values)
    
    for i in range(len(data[col])):
        data[col].iloc[i] = (data[col].iloc[i] - min_val) / (max_val - min_val)
    
    return data

def not_number_columns(data):
    '''
    Returns list of not number columns
    '''
    tmp_list = []
    for i in data:
        if data[i].dtypes != float and data[i].dtypes != int:
            tmp_list.append(i)
    return tmp_list

def nan_columns(data):
    '''
    Returns list of nan-containing columns
    '''
    tmp_list = []
    for i in data:
        if False in data[i].notna().values:
            tmp_list.append(i)
    return tmp_list#, asa