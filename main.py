from helpfunction import load_data_normalise, SeqDataset, plot_loss, plot_predictions, bit_to_hyperparameters, eval_model, train_batch_ind, k_fold_datav2, plot_average_predictionsv2, basis_func
from helpfunction import load_data_normalise, SeqDataset, plot_loss, plot_predictions, bit_to_hyperparameters, eval_model, train_batch_ind, k_fold_datav2, plot_average_predictionsv2, basis_func
from ParametricLSTMCNN import ParametricLSTMCNN
import torch
import numpy as np


### data split using individual batteries and trained to flip between batteries
verbose = True
battery = ['B0005', 'B0006', 'B0007', 'B0018']

model_type = 'data'
n_epoch = 100

test_size = 0.2
cv_size = 0.2

bit = [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 2, 1, 2, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 1, 0, 2, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 2, 0, 1, 0, 0, 2, 1, 0, 0, 0, 2, 0, 0, 0, 0, 1, 2, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1]
seq_length, num_layers_conv, output_channels, kernel_sizes, stride_sizes, padding_sizes, hidden_size_lstm, num_layers_lstm, hidden_neurons_dense, lr, batch_size, n_epoch = [12,
                                        3, [12, 18, 4], [3, 3, 3], [1, 1, 1], [1, 1, 1], 20, 1, [40, 12], 0.001, 64, 20]

if verbose:
    print(f'model type is {model_type}')
if model_type == 'hybrid':
    inputlstm = 5
elif model_type == 'data':
    inputlstm = 4

# model initialisation
model = ParametricLSTMCNN(num_layers_conv, output_channels, kernel_sizes, stride_sizes, padding_sizes, hidden_size_lstm, num_layers_lstm, hidden_neurons_dense, seq_length, inputlstm)
lf = torch.nn.MSELoss()
opimiser = torch.optim.Adam(model.parameters(), lr=lr)
torch.manual_seed(0)
np.random.seed(121)
# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device is {device}')
model.to(device)


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
    print(f'train batteries are {train_battery[0]} and {train_battery[1]}')
    
    normalised_data, data_max, data_min, size_of_bat_train = load_data_normalise(train_battery, model_type)
    normalised_data_val, _, _, size_of_bat_val = load_data_normalise(val_battery, model_type)
    normalised_data_test, _, _, size_of_bat_test = load_data_normalise(test_battery, model_type)

    x_train_bat, y_train_bat = k_fold_datav2(normalised_data, seq_length=seq_length, model_type=model_type, size_of_bat=size_of_bat_train)
    x_val, y_val = k_fold_datav2(normalised_data_val, seq_length=seq_length, model_type=model_type, size_of_bat=size_of_bat_val)
    x_test, y_test = k_fold_datav2(normalised_data_test, seq_length=seq_length, model_type=model_type, size_of_bat=size_of_bat_test)

    trainloader = SeqDataset(x_train_bat, y_data=y_train_bat, seq_len=seq_length, batch=batch_size)
    val_loader = SeqDataset(x_val, y_data=y_val, seq_len=seq_length, batch=batch_size)
    test_loader = SeqDataset(x_test, y_data=y_test, seq_len=seq_length, batch=batch_size)
    
    # Training model
    model, train_loss_history, val_loss_history = train_batch_ind(model, trainloader, val_loader, n_epoch=n_epoch, lf=lf, optimiser=opimiser, verbose=True)

    # Evaluation
    # eval_model(model, x_test, y_test, criterion=lf)

    # Plotting
    # if i == 3:
    plot_loss(train_loss_history, val_loss_history)
    plot_predictions(model, test_dataloader=test_loader, data_max=data_max, data_min = data_min, model_type=model_type, batch_size=batch_size)
        # plot_average_predictionsv2(model, x_test, y_test, ttd_mean=mean_ttd, ttd_std=std_ttd, model_type=model_type)
