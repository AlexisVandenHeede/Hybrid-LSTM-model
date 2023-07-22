# import torch
# print(f'PyTorch version: {torch.__version__}')
# print('*'*10)
# print(f'_CUDA version: ')
# print('*'*10)
# print(f'CUDNN version: {torch.backends.cudnn.version()}')
# print(f'Available GPU devices: {torch.cuda.device_count()}')
# print(f'Device Name: {torch.cuda.get_device_name()}')
import pandas as pd
# import numpy as np
battery = ['B0005']
# data = pd.read_csv(f'data/padded_data_hybrid_{battery}.csv')
# data.drop('index', axis=1, inplace=True)
# data.drop('Unnamed: 0', axis=1, inplace=True)
# data.to_csv(f'data/padded_data_hybrid_{battery}.csv')

data = pd.read_csv("data/B0005_TTD - with SOC.csv")
data_padded = pd.read_csv(f'data/padded_data_hybrid_{battery}.csv')
print(len(data_padded['Voltage']))
print(len(data_padded['TTD']))
# from helpfunction import create_time_padding

# ['B0006', 'B0007', 'B0018'],
# create_time_padding(['B0005'], 'hybrid', 10)
"""
def create_time_padding(battery, model_type, n):
    '''
    Will time pad sawtooth functions with n data points before and after.
    '''
    data = []
    if model_type == 'data':
        for i in battery:
            data.append(pd.read_csv("data/" + i + "_TTD1.csv"))
    elif model_type == 'hybrid':
        for i in battery:
            data.append(pd.read_csv("data/" + i + "_TTD - with SOC.csv"))
    TTD = data('TTD')
    index_jumps = TTD.where(TTD == 0, 1)
    new_cycle = TTD.where(TTD.diff() < 0, 1)
    new_cycle[0] = 1
    new_cycle[new_cycle != 1] = 0
    new_cycle *= n
    index_jumps = index_jumps.replace({0:1, 1:0}) * n
    # print(f'index_jumps = {index_jumps}')
    # print(data)
    new_data = data.index.repeat(index_jumps)
    # print(new_data)
    new_data = pd.concat([data, data.iloc[new_data]])
    new_data_1 = data.index.repeat(new_cycle)
    new_data = pd.concat([new_data, data.iloc[new_data_1]])
    # print(new_data)
    new_data.sort_index(inplace=True)
    return new_data.reset_index()

n = 3
data = pd.read_csv('testdata.csv')
TTD = data['TTD']
index_jumps = TTD.where(TTD == 0, 1)
new_cycle = TTD.where(TTD.diff() < 0, 1)
new_cycle[0] = 1
new_cycle[new_cycle != 1] = 0
new_cycle *= n
index_jumps = index_jumps.replace({0:1, 1:0}) * n
# print(f'index_jumps = {index_jumps}')
# print(data)
new_data = data.index.repeat(index_jumps)
# print(new_data)
new_data = pd.concat([data, data.iloc[new_data]])
new_data_1 = data.index.repeat(new_cycle)
new_data = pd.concat([new_data, data.iloc[new_data_1]])
# print(new_data)
new_data.sort_index(inplace=True)
print(new_data.reset_index())
'''
for i in range(len(index_jumps)):
    zero_index = index_jumps[i]
    if i <= len(index_jumps) - 1:
        next_zero_index = index_jumps[i]
        last_rows.append(data.iloc[zero_index:next_zero_index+1].drop('time', axis=1,))
        first_rows.append(data.iloc[zero_index+1:next_zero_index+2].drop('time', axis=1))
        data.iloc[next_zero_index+1:, 0] += n
        index_jumps[i] += n
data.index = data['time']

data.drop('time', axis=1, inplace=True)
first_rows.pop()  # remove last element as it is empty
print(f' Updated index_jumps = {index_jumps}')
print(f'last rows: {last_rows}')
print(f'first rows: {first_rows}')
# for i in index_jumps:
#     print(f'i = {i}')
#     for j in range(1, n+1):
#         data = pd.concat([data, data.iloc[i:i+1].rename(index={i: i+j})])
print(data)
for i in range(len(index_jumps)):
    print(f'type to be added: {type(first_rows[i])}')
    to_be_added = first_rows[i]*3
    print(f'to be added: {to_be_added}')
    for j in range(1, n+1):
        data = pd.concat([data, last_rows[i].rename(index={index_jumps[i]: index_jumps[i]+j})])
data.sort_index(inplace=True)
print(data)
'''

"""