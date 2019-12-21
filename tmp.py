import pandas as pd

train_data = pd.read_csv('data/train.csv', index_col=False)
tmp_list = []
for i in train_data['CTR_CATEGO_X']:
    if i not in tmp_list:
        print(i)
        tmp_list.append(i)

print(tmp_list)

print(train_data.dropna())
#print(train_data['CTR_CATEGO_X'])