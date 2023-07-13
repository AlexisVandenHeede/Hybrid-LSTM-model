from helpfunction import load_data_normalise, data_split, SeqDataset, train_batch
from ParametricLSTMCNN import ParametricLSTMCNN
import torch

verbose = True
battery = ['B0006', 'B0007', 'B0018']
model_type = 'data'
n_epoch = 5
test_size = 0.1
cv_size = 0.1
lr = 0.001
seq_length = 50
batch_size = 600
if verbose:
    print(f'model type is {model_type}')
normalised_data, time_mean, time_std = load_data_normalise(battery, model_type)

# data initialisation
X_train, y_train, X_test, y_test, X_val, y_val = data_split(normalised_data, test_size=test_size, cv_size=cv_size, seq_length=seq_length)
# hyperparameters
num_layers_conv = 1
output_channels = [6]
kernel_sizes = [4]
stride_sizes = [2]
padding_sizes = [4]
hidden_size_lstm = 10
num_layers_lstm = 3
hidden_neurons_dense = [4, 1]
inputlstm = X_train.shape[2]
# model intisisation
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
# do i need test dataset not on other file
# test_dataset = SeqDataset(x_data=X_test, y_data=y_test, seq_leng=seq_length, batch_size=batch_size)

"""
train_dataset.to(device)
validation_dataset.to(device)
# test_dataset.to(device)
"""

# Training model
model, train_loss_history, val_loss_history = train_batch(model, train_dataset, validation_dataset, n_epoch=n_epoch, lf=lf, optimiser=opimiser, verbose=True)
