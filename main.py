from helpfunction import load_data_normalise, data_split
from ParametricLSTMCNN import ParametricLSTMCNN
import torch

verbose = True
battery = ['B0006', 'B0007', 'B0018']
model_type = 'data'
if verbose:
    print(f'model type is {model_type}')
normalised_data, time_mean, time_std = load_data_normalise(battery, model_type)

# hyberparameters
hyperparameters = 1
# model intisisation
model = ParametricLSTMCNN(hyperparameters)
# data initialisation
X_train, y_train, X_test, y_test, X_val, y_val = data_split(normalised_data, test_size=0.2, cv_size=0.2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_train = torch.tensor(X_train.values).float().to(device)
y_train = torch.tensor(y_train.values).float().to(device)
X_test = torch.tensor(X_test.values).float().to(device)
y_test = torch.tensor(y_test.values).float().to(device)
X_val = torch.tensor(X_val.values).float().to(device)
y_val = torch.tensor(y_val.values).float().to(device)
