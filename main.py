from helpfunction import load_data_normalise_indv2, SeqDataset, plot_loss, plot_predictions, bit_to_hyperparameters, eval_model, train_batch_ind, k_fold_datav2, plot_average_predictionsv2, basis_func
from helpfunction import load_data_normalise_indv2, SeqDataset, plot_loss, plot_predictions, bit_to_hyperparameters, eval_model, train_batch_ind, k_fold_datav2, plot_average_predictionsv2, basis_func, seq_split
from ParametricLSTMCNN import ParametricLSTMCNN
import torch
import numpy as np


# everything that was here before - idk if it's needed
# verbose = True
# battery = ['B0005', 'B0006', 'B0007', 'B0018']
# model_type = 'data_padded'
# n_epoch = 100
# test_size = 0.1
# cv_size = 0.1
# # data driven padded
# # [0, 1, 0, 0, 0, 2, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 2, 0, 0, 0, 1, 1, 0, 0, 2, 1, 1, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 2, 1, 0, 0, 2, 2, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 2, 1, 1, 1, 0, 2, 0, 0, 0, 0]
# # rmse = 0.5094981116794153
# #  [0, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 1, 2, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 2, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 2, 2, 2]
# # 0.35047935866891333
# #  [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 2, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 2, 0, 0, 1, 0, 0, 1, 0, 0, 0, 2, 1, 1, 0, 2, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 1, 0, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 0, 0, 1]
# #  0.3780989659138594
# bit = [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 2, 1, 2, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 1, 0, 2, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 2, 0, 1, 0, 0, 2, 1, 0, 0, 0, 2, 0, 0, 0, 0, 1, 2, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1]
# #  0.31487230597996485
# # [0, 1, 0, 0, 2, 2, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 2, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 2, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 2, 2, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 2, 0, 0, 0, 1, 1, 0, 0]
# #  0.33249841991347645

# # hybrid padded
# # bit = [0, 1, 2, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 2, 0, 1, 0, 0, 1, 1, 1, 2, 0, 0, 0, 1, 2, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 2, 1, 0, 0, 1, 1, 0, 1, 0, 1, 2, 0, 0, 0, 2, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0]
# #  0.3842639607759947
# # [1, 1, 1, 1, 0, 0, 2, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 3, 2, 0, 0, 2, 1, 0, 0, 2, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 2, 2, 0, 0, 0, 0, 0, 0, 0]
# # Fitness =  0.44335262456618796
# seq_length, num_layers_conv, output_channels, kernel_sizes, stride_sizes, padding_sizes, hidden_size_lstm, num_layers_lstm, hidden_neurons_dense, lr, batch_size, n_epoch, hyperparameters = bit_to_hyperparameters(bit)
# # hyperparameters = [seq_length, num_layers_conv, output_channels, kernel_sizes, stride_sizes, padding_sizes, hidden_size_lstm, num_layers_lstm, hidden_neurons_dense, lr, batch_size, n_epoch]

# if verbose:
#     print(f'model type is {model_type}')
# normalised_data, time_mean, time_std = load_data_normalise(battery, model_type)
# # pad_data = create_time_padding(normalised_data, n=5)
# # print(f'data is paddded')
# # data initialisation
# X_train, y_train, X_test, y_test, X_val, y_val = data_split(normalised_data, test_size=test_size, cv_size=cv_size, seq_length=seq_length, model_type=model_type)
# # hyperparameters
# inputlstm = X_train.shape[2]

# # model initialisation
# model = ParametricLSTMCNN(num_layers_conv, output_channels, kernel_sizes, stride_sizes, padding_sizes, hidden_size_lstm, num_layers_lstm, hidden_neurons_dense, seq_length, inputlstm)
# lf = torch.nn.MSELoss()
# opimiser = torch.optim.Adam(model.parameters(), lr=lr)

# # device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f'device is {device}')
# model.to(device)

# # # data loader
# # train_dataset = SeqDataset(x_data=X_train, y_data=y_train, seq_len=seq_length, batch=batch_size)
# # validation_dataset = SeqDataset(x_data=X_val, y_data=y_val, seq_len=seq_length, batch=batch_size)

# """
# train_dataset.to(device)
# validation_dataset.to(device)
# # test_dataset.to(device)
# """

# # # Training model
# # model, train_loss_history, val_loss_history = train_batch(model, train_dataset, validation_dataset, n_epoch=n_epoch, lf=lf, optimiser=opimiser, verbose=True)
# # eval_model(model, X_test, y_test, criterion=lf)
# # plot_loss(train_loss_history, val_loss_history)
# # plot_predictions(model, X_test, y_test, ttd_mean=time_mean, ttd_std=time_std, model_type=model_type)




### data split using individual batteries and trained to flip between batteries
verbose = True
battery = ['B0005', 'B0006', 'B0007', 'B0018']
model_type = 'data_padded'
n_epoch = 100
test_size = 0.1
cv_size = 0.1
# some data-padded hyperparameters from ga
bit = [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 2, 1, 2, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 1, 0, 2, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 2, 0, 1, 0, 0, 2, 1, 0, 0, 0, 2, 0, 0, 0, 0, 1, 2, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1]
# [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 2, 0, 0, 1, 0, 2, 0, 1, 0, 2, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0]
# [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 2, 1, 2, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 1, 0, 2, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 2, 0, 1, 0, 0, 2, 1, 0, 0, 0, 2, 0, 0, 0, 0, 1, 2, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1]
seq_length, num_layers_conv, output_channels, kernel_sizes, stride_sizes, padding_sizes, hidden_size_lstm, num_layers_lstm, hidden_neurons_dense, lr, batch_size, n_epoch, hyperparameters = bit_to_hyperparameters(bit)
seq_steps = 5
if verbose:
    print(f'model type is {model_type}')
if model_type == 'hybrid_padded':
    inputlstm = 5
elif model_type == 'data_padded':
    inputlstm = 4

# model initialisation

for i in range(1):
    battery_temp = battery.copy()
    test_battery = [battery[i]]
    print(f'test battery is {test_battery}')
    battery_temp.remove(test_battery[0])
    if i == 3:
        val_battery = [battery[0]]
    else:
        val_battery = [battery[i+1]]
    battery_temp.remove(val_battery[0])
    print(f'validation battery is {val_battery}')
    train_battery = battery_temp
    print(f'train batteries are {train_battery} and {train_battery[1]}')
    
    normalised_data_bat, mean_train, std_train, size_of_bat_train = load_data_normalise_indv2(train_battery, model_type)
    normalised_data_val, mean_val, std_val, size_of_bat_val = load_data_normalise_indv2(val_battery, model_type)
    normalised_data_test, mean_test, std_test, size_of_bat_test = load_data_normalise_indv2(test_battery, model_type)
    
    x_train_bat_1, y_train_bat_1, x_train_bat_2, y_train_bat_2 = seq_split(train_battery, normalised_data_bat, mean_train, std_train, seq_steps=seq_steps, model_type=model_type)
    x_val, y_val = seq_split(val_battery, normalised_data_val, mean_val, std_val, seq_steps=seq_steps, model_type=model_type)
    x_test, y_test = seq_split(test_battery, normalised_data_test, mean_test, std_test, seq_steps=seq_steps, model_type=model_type)

    trainloader_1 = SeqDataset(x_train_bat_1, y_data=y_train_bat_1, batch=batch_size)
    # trainloader_2 = SeqDataset(x_train_bat_2, y_data=y_train_bat_2, batch=batch_size)
    val_loader = SeqDataset(x_val, y_data=y_val, batch=batch_size)

    # # Training model
    seq_length = x_train_bat_1.shape[1]
    model = ParametricLSTMCNN(num_layers_conv, output_channels, kernel_sizes, stride_sizes, padding_sizes, hidden_size_lstm, num_layers_lstm, hidden_neurons_dense, seq_length, inputlstm)
    lf = torch.nn.MSELoss()
    opimiser = torch.optim.Adam(model.parameters(), lr=lr)
    torch.manual_seed(0)
    np.random.seed(121)

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device is {device}')
    model.to(device)

    model, train_loss_history, val_loss_history = train_batch_ind(model, trainloader_1, val_loader, n_epoch=n_epoch, lf=lf, optimiser=opimiser, verbose=True)

    # Evaluation
    eval_model(model, x_test, y_test, criterion=lf)

    # Plotting
    if i == 0:
        plot_loss(train_loss_history, val_loss_history)
        plot_predictions(model, x_test, y_test, ttd_mean=mean_test, ttd_std=std_test, model_type=model_type)
        plot_average_predictionsv2(model, x_test, y_test, ttd_mean=mean_test, ttd_std=std_test, model_type=model_type)
