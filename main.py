from helpfunction import load_data_normalise, data_split, SeqDataset, train_batch, plot_loss, plot_predictions, bit_to_hyperparameters, eval_model
from ParametricLSTMCNN import ParametricLSTMCNN
import torch

verbose = True
battery = ['B0005', 'B0006', 'B0007', 'B0018']
model_type = 'hybrid_padded'
n_epoch = 100
test_size = 0.1
cv_size = 0.1
lr = 0.024
seq_length = 25
batch_size = 362
if verbose:
    print(f'model type is {model_type}')
normalised_data, time_mean, time_std = load_data_normalise(battery, model_type)
# pad_data = create_time_padding(normalised_data, n=5)
# print(f'data is paddded')
# data initialisation
X_train, y_train, X_test, y_test, X_val, y_val = data_split(normalised_data, test_size=test_size, cv_size=cv_size, seq_length=seq_length, model_type=model_type)
# hyperparameters
num_layers_conv = 1
output_channels = [1]
kernel_sizes = [1]
stride_sizes = [1]
padding_sizes = [1]
hidden_size_lstm = 3
num_layers_lstm = 9
hidden_neurons_dense = [1, 1]
inputlstm = 7
# data driven padded
# [0, 1, 0, 0, 0, 2, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 2, 0, 0, 0, 1, 1, 0, 0, 2, 1, 1, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 2, 1, 0, 0, 2, 2, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 2, 1, 1, 1, 0, 2, 0, 0, 0, 0]
# rmse = 0.5094981116794153
#  [0, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 1, 2, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 2, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 2, 2, 2]
# 0.35047935866891333
#  [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 2, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 2, 0, 0, 1, 0, 0, 1, 0, 0, 0, 2, 1, 1, 0, 2, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 1, 0, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 0, 0, 1]
#  0.3780989659138594
# bit = [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 2, 1, 2, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 1, 0, 2, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 2, 0, 1, 0, 0, 2, 1, 0, 0, 0, 2, 0, 0, 0, 0, 1, 2, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1]
#  0.31487230597996485
# [0, 1, 0, 0, 2, 2, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 2, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 2, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 2, 2, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 2, 0, 0, 0, 1, 1, 0, 0]
#  0.33249841991347645
# hybrid padded
# bit = [0, 1, 2, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 2, 0, 1, 0, 0, 1, 1, 1, 2, 0, 0, 0, 1, 2, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 2, 1, 0, 0, 1, 1, 0, 1, 0, 1, 2, 0, 0, 0, 2, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0]
#  0.3842639607759947
# hyperparameters = bit_to_hyperparameters(bit)
# hyperparameters = [seq_length, num_layers_conv, output_channels, kernel_sizes, stride_sizes, padding_sizes, hidden_size_lstm, num_layers_lstm, hidden_neurons_dense, lr, batch_size, n_epoch]
# model initialisation
model = ParametricLSTMCNN(num_layers_conv, output_channels, kernel_sizes, stride_sizes, padding_sizes, hidden_size_lstm, num_layers_lstm, hidden_neurons_dense, seq_length, inputlstm)
lf = torch.nn.MSELoss()
opimiser = torch.optim.Adam(model.parameters(), lr=lr)

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device is {device}')
model.to(device)

# data loader
train_dataset = SeqDataset(x_data=X_train, y_data=y_train, seq_len=seq_length, batch=batch_size)
validation_dataset = SeqDataset(x_data=X_val, y_data=y_val, seq_len=seq_length, batch=batch_size)

"""
train_dataset.to(device)
validation_dataset.to(device)
# test_dataset.to(device)
"""

# Training model
model, train_loss_history, val_loss_history = train_batch(model, train_dataset, validation_dataset, n_epoch=n_epoch, lf=lf, optimiser=opimiser, verbose=True)
eval_model(model, X_test, y_test, criterion=lf)
plot_loss(train_loss_history, val_loss_history)
plot_predictions(model, X_test, y_test, ttd_mean=time_mean, ttd_std=time_std, model_type=model_type)
