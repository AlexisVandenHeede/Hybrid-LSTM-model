from helpfunction import train_batch_ind, bit_to_hyperparameters, plot_average_predictions, load_data_normalise_ind, SeqDataset, k_fold_data, ParametricLSTMCNN, eval_model
import torch

battery = ['B0005', 'B0006', 'B0007', 'B0018']
# data driven padded
# [0, 1, 0, 0, 0, 2, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 2, 0, 0, 0, 1, 1, 0, 0, 2, 1, 1, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 2, 1, 0, 0, 2, 2, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 2, 1, 1, 1, 0, 2, 0, 0, 0, 0]
# rmse = 0.5094981116794153
#  [0, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 1, 2, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 2, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 2, 2, 2]
# 0.35047935866891333
#  [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 2, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 2, 0, 0, 1, 0, 0, 1, 0, 0, 0, 2, 1, 1, 0, 2, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 1, 0, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 0, 0, 1]
#  0.3780989659138594
bit = [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 2, 1, 2, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 1, 0, 2, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 2, 0, 1, 0, 0, 2, 1, 0, 0, 0, 2, 0, 0, 0, 0, 1, 2, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1]
#  0.31487230597996485
# [0, 1, 0, 0, 2, 2, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 2, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 2, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 2, 2, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 2, 0, 0, 0, 1, 1, 0, 0]
#  0.33249841991347645
# hybrid padded
# bit = [0, 1, 2, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 2, 0, 1, 0, 0, 1, 1, 1, 2, 0, 0, 0, 1, 2, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 2, 1, 0, 0, 1, 1, 0, 1, 0, 1, 2, 0, 0, 0, 2, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0]
#  0.3842639607759947
# [1, 1, 1, 1, 0, 0, 2, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 3, 2, 0, 0, 2, 1, 0, 0, 2, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 2, 2, 0, 0, 0, 0, 0, 0, 0]
# Fitness =  0.44335262456618796
seq_length, num_layers_conv, output_channels, kernel_sizes, stride_sizes, padding_sizes, hidden_size_lstm, num_layers_lstm, hidden_neurons_dense, lr, batch_size, n_epoch, hyperparameters = bit_to_hyperparameters(bit)
# kfold_ind(model_type='data_padded', hyperparameters=hyperparameters, battery=['B0005', 'B0006', 'B0007', 'B0018'], plot=True, strict=False)
model_type = 'data_padded'
if model_type == 'hybrid_padded':
    inputlstm = 7
elif model_type == 'data_padded':
    inputlstm = 6

# model initialisation
model = ParametricLSTMCNN(num_layers_conv, output_channels, kernel_sizes, stride_sizes, padding_sizes, hidden_size_lstm, num_layers_lstm, hidden_neurons_dense, seq_length, inputlstm)
lf = torch.nn.MSELoss()
opimiser = torch.optim.Adam(model.parameters(), lr=lr)

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device is {device}')
model.to(device)

for i in range(4):
    battery_temp = battery.copy()
    test_battery = battery[i]
    print(f'test battery is {test_battery}')
    battery_temp.remove(test_battery)
    if i == 3:
        val_battery = battery[0]
    else:
        val_battery = battery[i+1]
    battery_temp.remove(val_battery)
    print(f'validation battery is {val_battery}')
    train_battery_1 = battery_temp[0]
    train_battery_2 = battery_temp[1]
    print(f'train batteries are {train_battery_1} and {train_battery_2}')
    
    normalised_data_bat_1, _, _ = load_data_normalise_ind(train_battery_1, model_type)
    normalised_data_bat_2, _, _ = load_data_normalise_ind(train_battery_2, model_type)
    normalised_data_val, _, _ = load_data_normalise_ind(val_battery, model_type)
    normalised_data_test, mean_ttd, std_ttd = load_data_normalise_ind(test_battery, model_type)

    x_train_bat_1, y_train_bat_1 = k_fold_data(normalised_data_bat_1, seq_length=seq_length, model_type=model_type)
    x_train_bat_2, y_train_bat_2 = k_fold_data(normalised_data_bat_2, seq_length=seq_length, model_type=model_type)
    x_val, y_val = k_fold_data(normalised_data_bat_1, seq_length=seq_length, model_type=model_type)
    x_test, y_test = k_fold_data(normalised_data_test, seq_length=seq_length, model_type=model_type)

    # print(f'shapes of x_train_bat_1, x_train_bat_2, x_val, x_test are {x_train_bat_1.shape}, {x_train_bat_2.shape}, {x_val.shape}, {x_test.shape}')

    trainloader_1 = SeqDataset(x_train_bat_1, y_data=y_train_bat_1, seq_len=seq_length, batch=batch_size)
    trainloader_2 = SeqDataset(x_train_bat_2, y_data=y_train_bat_2, seq_len=seq_length, batch=batch_size)
    val_loader = SeqDataset(x_val, y_data=y_val, seq_len=seq_length, batch=batch_size)

    # Training model
    model, train_loss_history, val_loss_history = train_batch_ind(model, trainloader_1, trainloader_2, val_loader, n_epoch=n_epoch, lf=lf, optimiser=opimiser, verbose=True)

    # Evaluation
    eval_model(model, x_test, y_test, criterion=lf)

    plot_average_predictions(model, x_test, y_test, ttd_mean=mean_ttd, ttd_std=std_ttd, model_type=model_type)



# seq_length = 50
# num_layers_conv = 1
# output_channels = [6]
# kernel_sizes = [4]
# stride_sizes = [2]
# padding_sizes = [4]
# hidden_size_lstm = 10
# num_layers_lstm = 3
# hidden_neurons_dense = [4, 1]
# lr = 0.001
# batch_size = 600
# n_epoch = 50

# hyperparameters = [seq_length, num_layers_conv, output_channels, kernel_sizes, stride_sizes, padding_sizes, hidden_size_lstm, num_layers_lstm, hidden_neurons_dense, lr, batch_size, n_epoch]
# k_fold(model_type='data_padded', hyperparameters=hyperparameters, battery=['B0005', 'B0006', 'B0007', 'B0018'], verbose=False)


# import torch
# print(f'PyTorch version: {torch.__version__}')
# print('*'*10)
# print(f'_CUDA version: ')
# print('*'*10)
# print(f'CUDNN version: {torch.backends.cudnn.version()}')
# print(f'Available GPU devices: {torch.cuda.device_count()}')
# print(f'Device Name: {torch.cuda.get_device_name()}')
# import pandas as pd
# import numpy as np
# battery = ['B0005']
# data = pd.read_csv(f'data/padded_data_hybrid_{battery}.csv')
# data.drop('index', axis=1, inplace=True)
# data.drop('Unnamed: 0', axis=1, inplace=True)
# data.to_csv(f'data/padded_data_hybrid_{battery}.csv')

# data = pd.read_csv("data/B0005_TTD - with SOC.csv")
# data_padded = pd.read_csv(f'data/padded_data_hybrid_{battery}.csv')
# print(len(data_padded['Voltage']))
# print(len(data_padded['TTD']))
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
