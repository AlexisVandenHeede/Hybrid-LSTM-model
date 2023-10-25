from helpfunction import load_data_normalise_indv2, SeqDataset, plot_loss, plot_predictions, bit_to_hyperparameters, eval_model, train_batch_ind, k_fold_datav2, plot_average_predictionsv2, basis_func
from helpfunction import load_data_normalise_indv2, SeqDataset, plot_loss, plot_predictions, bit_to_hyperparameters, eval_model, train_batch_ind, k_fold_datav2, plot_average_predictionsv2, basis_func
from ParametricLSTMCNN import ParametricLSTMCNN, TwoLayerCNNLSTM, TwoLayerCNNLSTM1
import torch
import numpy as np

### data split using individual batteries and trained to flip between batteries
verbose = True
battery = ['B0005', 'B0006', 'B0007', 'B0018']
model_type = 'data'
n_epoch = 20
test_size = 0.1
cv_size = 0.1
# some data-padded hyperparameters from ga
bit = [0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0]
bit = [0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1]

print(bit_to_hyperparameters(bit))
#yo = [18, 1, [43], [3], [1], [1], 4, 1, 0, 0.04515, 1833, 100]
#seq_length, num_layers_conv, num_kernels, kernel_sizes, stride_sizes, padding_sizes, hidden_size_lstm, num_layers_lstm, hidden_neurons_dense, lr, batch_size, n_epoch = yo#bit_to_hyperparameters(bit) #[50, 1, [4], [3], [1], [1], 10, 1, 10, 0.0005, 1000, 100]

if verbose:
    print(f'model type is {model_type}')
if model_type == 'hybrid_padded':
    inputlstm = 5
elif model_type == 'data_padded' :
    inputlstm = 4

inputlstm = 4

# inputlstm changed meaning, should optimise too
# model initialisation
#model = ParametricLSTMCNN(num_layers_conv, num_kernels, kernel_sizes, stride_sizes, padding_sizes, hidden_size_lstm, num_layers_lstm, hidden_neurons_dense, seq_length, inputlstm, batch_size)

num_kernels, kernel_sizes, stride_sizes, padding_sizes, hidden_size_lstm, num_layers_lstm, slope, seq_length, inputlstm, lr, batch_size, dense_neurons = [20, [3, 3], [1, 1], [1, 1], 21, 1, 0.1, 20, 4, 0.2, 801, 30]#bit_to_hyperparameters(bit) #[10, [3, 3], [1, 1], [1, 1], 30, 2, 0.2, 50, 4, 0.0005, 1000, 15]
#[39, [3, 3], [1, 1], [1, 1], 12, 3, 2.51, 21, 4, 0.07845, 483, 6]
model = TwoLayerCNNLSTM(num_kernels, kernel_sizes, stride_sizes, padding_sizes, hidden_size_lstm, num_layers_lstm, slope, seq_length, inputlstm, dense_neurons)

lf = torch.nn.MSELoss()
opimiser = torch.optim.Adam(model.parameters(), lr=lr)
torch.manual_seed(0)
np.random.seed(0)
# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device is {device}')
model.to(device)


for i in range(4):
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
    
    normalised_data_bat, _, _, size_of_bat_train = load_data_normalise_indv2(train_battery, model_type)
    normalised_data_val, _, _, size_of_bat_val = load_data_normalise_indv2(val_battery, model_type)
    normalised_data_test, mean_ttd, std_ttd, size_of_bat_test = load_data_normalise_indv2(test_battery, model_type)

    
    x_train_bat, y_train_bat = k_fold_datav2(normalised_data_bat, seq_length=seq_length, model_type=model_type, size_of_bat=size_of_bat_train)
    x_val, y_val = k_fold_datav2(normalised_data_val, seq_length=seq_length, model_type=model_type, size_of_bat=size_of_bat_val)
    x_test, y_test = k_fold_datav2(normalised_data_test, seq_length=seq_length, model_type=model_type, size_of_bat=size_of_bat_test)

    trainloader = SeqDataset(x_train_bat, y_data=y_train_bat, seq_len=seq_length, batch=batch_size)
    val_loader = SeqDataset(x_val, y_data=y_val, seq_len=seq_length, batch=batch_size)

    # Training model
    model, train_loss_history, val_loss_history = train_batch_ind(model, trainloader, val_loader, n_epoch=n_epoch, lf=lf, optimiser=opimiser, verbose=True)

    # Evaluation
    eval_model(model, x_test, y_test, criterion=lf)

    # Plotting
    #if i == 3:
    plot_loss(train_loss_history, val_loss_history)
    plot_predictions(model, x_test, y_test, ttd_mean=mean_ttd, ttd_std=std_ttd, model_type=model_type)
    plot_average_predictionsv2(model, x_test, y_test, ttd_mean=mean_ttd, ttd_std=std_ttd, model_type=model_type)
