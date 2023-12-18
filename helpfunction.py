import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import torch
from ParametricLSTMCNN import ParametricLSTMCNN
from bitstring import BitArray


def load_data_normalise(battery, model_type):
    debug = False
    """
    Load the data and normalize it.
    Return: normalized data, mean time, std time
    """
    data = []
    size_of_bat = []
    if model_type not in ['data', 'hybrid', 'data_padded', 'hybrid_padded']:
        print('Wrong model type, either data, hybrid, data_padded, or hybrid_padded')
        raise ValueError

    for i in battery:
        data_file = f"data/{i}_TTD1.csv" if model_type == 'data' else f"data/{i}_TTD - with SOC.csv" if model_type == 'hybrid' else f"data/padded_data_mod_volt[{i}].csv" if model_type == 'data_padded' else f"data/padded_data_hybrid_w_ecm[{i}].csv"
        battery_data = pd.read_csv(data_file)
        time = battery_data['TTD']
        
        data_max = battery_data.mean(axis=0)
        data_min = battery_data.std(axis=0)
        
        normalized_battery_data = (battery_data - data_min) / (data_max - data_min)
        normalized_battery_data = normalized_battery_data.astype('float16')  # Convert to float16 to save memory
        data.append(normalized_battery_data)
        size_of_bat.append(len(battery_data))

    data = pd.concat(data)
    if debug:
        print(data.columns)
        
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    data = data.loc[:, ~data.columns.str.contains('^Current')]
    
    if debug:
        print(data.columns)

    # if debug:
    #     # Plot each normalized data
    #     thing = input('Press enter to see scatter plots of normalized data')
    #     if thing == '':
    #         for col in data.columns:
    #             plt.figure(figsize=(8, 6))
    #             plt.scatter(range(len(data)), data[col], s=5)
    #             plt.title(f'Scatter Plot for {col}')
    #             plt.xlabel('Data Point Index')
    #             plt.ylabel('Normalized Value')
    #             plt.grid(True)
    #             plt.show()

    return data, data_max, data_min, size_of_bat


class EarlyStopper:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        """Implement the early stopping criterion.
        The function has to return 'True' if the current validation loss (in the arguments) has increased
        with respect to the minimum value of more than 'min_delta' and for more than 'patience' steps.
        Otherwise the function returns 'False'.
        counter_1 is so that if no learning takes place return True
        """
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            return False
        else:
            if (validation_loss - self.min_validation_loss) >= -self.min_delta:
                self.counter += 1
            if self.counter >= self.patience:
                return True
            else:
                return False


def basis_func(scaling_factor, hidden_layers):
    """ Rescale hyperparameter per layer using basis function, now just np.arange"""
    basis = (np.arange(hidden_layers, dtype=int))*scaling_factor
    if hidden_layers == 1:
        basis[0] = 1
    basis_function = []
    for i in range(hidden_layers):
        if basis[i] == 0:
            basis[i] = 1
        basis_function.append(basis[i])
    return basis_function


def train_batch_ind(model, train_dataloader, val_dataloader, n_epoch, lf, optimiser, verbose = False):
    """
    train model dataloaders, early stopper Class
    """
    epoch = []
    early_stopper = EarlyStopper(patience=10, min_delta=0.001)
    with torch.no_grad():
        train_loss_history = []
        val_loss_history = []

    for i in range(n_epoch):
        loss_train = 0
        loss_val = 0
        model.train()
        epoch.append(i+1)
        
        batch_idx = 0
        for l, (x, y) in enumerate(train_dataloader):
            y = y.unsqueeze(1)
            batch_idx += 1
            target_train = model(x)
            loss = lf(target_train, y)
            loss_train += loss.item()
            
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            
        train_loss = loss_train/len(train_dataloader)
        model.eval() 
        for k, (x, y) in enumerate(val_dataloader):
            y = y.unsqueeze(1)
            target_val = model(x)
            loss = lf(target_val, y)
            loss_val += loss.item()

        val_loss = loss_val/len(val_dataloader)
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        
        if verbose:
            print(f"Epoch {i+1}: train loss = {train_loss:.10f}, val loss = {val_loss:.10f}")
        
        if early_stopper.early_stop(val_loss):
            print("Early stopping")
            break
  

    return model, train_loss_history, val_loss_history


def plot_loss(train_loss_history, val_loss_history):
    plt.plot(train_loss_history, label='train loss')
    plt.plot(val_loss_history, label='val loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()


def plot_predictions(model, test_dataloader, data_max, data_min, model_type, batch_size):
    loss_test = 0
    lf = torch.nn.MSELoss()
    y_pred = []
    y_test = []
    for k, (x, y) in enumerate(test_dataloader):
        model.eval()
        y = y.unsqueeze(1)
        target_val = model(x)
        
        if y.shape[0] != batch_size:
            break
        loss = lf(target_val, y)
        
        y_test.append(y.detach().numpy())
        y_pred.append(target_val.detach().numpy())
        
        loss_test += loss.item()
        
    test_loss = loss_test/len(test_dataloader)
    
    y_test = np.reshape(y_test, -1)
    y_test = y_test*(data_max['TTD'] - data_min['TTD']) + data_min['TTD']
    
    y_pred = np.reshape(y_pred, -1)
    y_pred = y_pred*(data_max['TTD'] - data_min['TTD']) + data_min['TTD']
    
    
    plt.plot(y_test, label='Actual')
    plt.plot(y_pred, label='Prediction')
    plt.xlabel('Time')
    plt.ylabel('TTD')
    plt.legend()
    plt.title(f'Predictions vs Actual for {model_type} model')
    plt.show()


def plot_average_predictionsv2(model, X_test, y_test, data_max, data_min):
    df = pd.DataFrame()
    predictions = model(X_test)
    df['predictions'] = np.reshape(predictions.cpu().detach().numpy(), -1) * (data_max['TTD'] - data_min['TTD']) + data_min['TTD']
    df.insert(1, 'y_test', np.reshape(y_test.cpu().detach().numpy(), -1) * (data_max['TTD'] - data_min['TTD']) + data_min['TTD'])
  
    threshold = 2000
    df['cycle_starts'] = df['y_test'].diff().abs() > threshold
    df['cycle_index'] = 0
    cycle_count = 0
    cycle_lengths = []

    for index, row in df.iterrows():
        if row['cycle_starts']:
            if cycle_count > 0:
                cycle_lengths.append(cycle_count)
            cycle_count = 0
        df.at[index, 'cycle_index'] = cycle_count
        cycle_count += 1
   
    # Calculate scaled_cycle_index
    df['scaled_cycle_index'] = 0
    current_cycle_start = 0
    for length in cycle_lengths:
        scaled_indices = np.arange(length) / length
        df.loc[current_cycle_start:current_cycle_start + length - 1, 'scaled_cycle_index'] = scaled_indices
        current_cycle_start += length

    #   round scaled_cycle_index to 2dp
    df['scaled_cycle_index'] = df['scaled_cycle_index'].round(2)

    # Calculate average values per scaled_cycle_index
    avg_scaled_values = df.groupby('scaled_cycle_index').agg({'predictions': 'mean', 'y_test': 'mean'}).reset_index()
    # remove first point
    avg_scaled_values = avg_scaled_values.iloc[1:]
    # Create a scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(avg_scaled_values['scaled_cycle_index'], avg_scaled_values['predictions'], label='Average Predictions', marker='o')
    plt.scatter(avg_scaled_values['scaled_cycle_index'], avg_scaled_values['y_test'], label='Average y_test', marker='x')
    plt.xlabel('Scaled Cycle Index')
    plt.ylabel('Average Value')
    plt.legend()
    plt.title('Average Predictions and Average y_test per Scaled Cycle Index')
    plt.grid(True)

    plt.show()



def create_sequence(x_data, y_data, seq_len):
    x_data = x_data.to_numpy()
    y_data = y_data.to_numpy()
    
    xs, ys = [], []
    
    for idx in range(len(x_data)):

        start_idx = idx * seq_len
        end_idx = start_idx + seq_len
               

        x = x_data[start_idx:end_idx]
        y = y_data[start_idx:end_idx]        
        if end_idx > len(x_data):
            break
        if x.shape[0] != seq_len:
            break
        if y.shape[0] != seq_len:
            break
        if x.shape[1] != x_data.shape[1]:
            break        
        xs.append(x)
        ys.append(y)

    xs = np.array(xs)
    
    ys = np.array(ys)
    xs = torch.from_numpy(xs).float()
    ys = torch.from_numpy(ys).float()
        
    return xs, ys
    
def SeqDataset(x_data, y_data, seq_len, batch):
    """
    create a sequence dataset
    """
    x, y = create_sequence(x_data, y_data, seq_len)
    dataset = torch.utils.data.TensorDataset(x, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=False)
    return dataloader


def eval_model(model, X_test, y_test, criterion):
    model.eval()
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()
    
    with torch.no_grad():
        y_pred = model(X_test)
        rmse = torch.sqrt(criterion(y_test, y_pred)).item()
        raw_test = ((y_test - y_pred) ** 2).mean().item()

    print(f'rmse_test = {rmse}')
    return rmse, raw_test


def k_fold_datav2(normalised_data, seq_length, model_type, size_of_bat):
    if model_type == 'data_padded' or model_type == 'data':
        X = normalised_data.drop(['TTD', 'Time', 'Start_time'], axis=1)
    elif model_type == 'hybrid_padded':
        X = normalised_data.drop(['TTD', 'Time', 'Start_time', 'Instance', 'Voltage_measured'], axis=1)
    y = normalised_data['TTD']
    

    return X, y


def kfold_ind(model_type, hyperparameters, battery, plot=False, strict=True):
    print(f'model type = {model_type}')
    k_fold_rmse = []
    k_fold_raw_test = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for i in range(4):
        battery_temp = battery.copy()
        test_battery = [battery[i]]
        print(f'test battery is {test_battery}')
        battery_temp.remove(test_battery[0])
        if i == 3:
            validation_battery = [battery[0]]
        else:
            validation_battery = [battery[i+1]]
        battery_temp.remove(validation_battery[0])
        print(f'validation battery is {validation_battery}')
        train_battery = battery_temp
        print(f'train batteries are {train_battery}')
        normalised_data_train, _, _, size_of_bat = load_data_normalise(train_battery, model_type)
        normalised_data_test, time_mean_test, time_std_test, size_of_bat_test = load_data_normalise(test_battery, model_type)
        normalised_data_validation, _, _, size_of_bat_val = load_data_normalise(validation_battery, model_type)
        seq_length = hyperparameters[0]
        X_train, y_train = k_fold_datav2(normalised_data_train, seq_length, model_type, size_of_bat)
        X_test, y_test = k_fold_datav2(normalised_data_test, seq_length, model_type, size_of_bat_test)
        X_validation, y_validation = k_fold_datav2(normalised_data_validation, seq_length, model_type, size_of_bat_val)
        model = ParametricLSTMCNN(hyperparameters[1], hyperparameters[2], hyperparameters[3], hyperparameters[4], hyperparameters[5], hyperparameters[6], hyperparameters[7], hyperparameters[8], hyperparameters[0], X_train.shape[2])
        lf = torch.nn.MSELoss()
        opimiser = torch.optim.Adam(model.parameters(), lr=hyperparameters[9])
        model.to(device)
        train_dataset = SeqDataset(x_data=X_train, y_data=y_train, seq_len=seq_length, batch=hyperparameters[10])
        validation_dataset = SeqDataset(x_data=X_validation, y_data=y_validation, seq_len=seq_length, batch=hyperparameters[10])
        model, train_loss_history, val_loss_history = train_batch_ind(model, train_dataset, validation_dataset, n_epoch=hyperparameters[11], lf=lf, optimiser=opimiser, verbose=True)
        rmse_test, raw_test = eval_model(model, X_test, y_test, lf)
        print(f'rmse_test = {rmse_test}')
        if plot:
            plot_loss(train_loss_history, val_loss_history)
            plot_predictions(model, X_test, y_test, time_mean_test, time_std_test, model_type)
        k_fold_rmse.append(rmse_test)
        k_fold_raw_test.append(raw_test)
        if strict:
            if np.mean(k_fold_rmse) > 1:
                print(f'average = {np.mean(k_fold_rmse)}')
                print('rmse too high')
                k_fold_rmse = 100
                break
    rmse_test = np.mean(k_fold_rmse)
    raw_test = np.mean(k_fold_raw_test)
    print(f'average rmse_test = {rmse_test} and raw_err = {raw_test}')
    return rmse_test


def bit_to_hyperparameters(bit):
    gene_length = 8
    n_epoch = 100

    seq_length = BitArray(bit[0:gene_length])
    num_layers_conv = BitArray(bit[gene_length:2*gene_length])
    output_channels = BitArray(bit[2*gene_length:3*gene_length])
    kernel_sizes = BitArray(bit[3*gene_length:4*gene_length])
    stride_sizes = BitArray(bit[4*gene_length:5*gene_length])
    padding_sizes = BitArray(bit[5*gene_length:6*gene_length])
    hidden_size_lstm = BitArray(bit[6*gene_length:7*gene_length])
    num_layers_lstm = BitArray(bit[7*gene_length:8*gene_length])
    hidden_neurons_dense = BitArray(bit[8*gene_length:9*gene_length])
    lr = BitArray(bit[9*gene_length:10*gene_length])
    batch_size = BitArray(bit[10*gene_length:11*gene_length])

    seq_length = seq_length.uint
    num_layers_conv = num_layers_conv.uint
    output_channels = output_channels.uint
    kernel_sizes = kernel_sizes.uint
    stride_sizes = stride_sizes.uint
    padding_sizes = padding_sizes.uint
    hidden_size_lstm = hidden_size_lstm.uint
    num_layers_lstm = num_layers_lstm.uint
    hidden_neurons_dense = hidden_neurons_dense.uint
    lr = lr.uint
    batch_size = batch_size.uint

    # resize hyperparameterss to be within range
    seq_length = int(np.interp(seq_length, [0, 255], [1, 50]))
    num_layers_conv = int(np.interp(num_layers_conv, [0, 255], [1, 10]))
    output_channels = int(np.interp(output_channels, [0, 255], [1, 10]))
    kernel_sizes = int(np.interp(kernel_sizes, [0, 255], [1, 10]))
    stride_sizes = int(np.interp(stride_sizes, [0, 255], [1, 10]))
    padding_sizes = int(np.interp(padding_sizes, [0, 255], [1, 10]))
    hidden_size_lstm = int(np.interp(hidden_size_lstm, [0, 255], [1, 10]))
    num_layers_lstm = int(np.interp(num_layers_lstm, [0, 255], [1, 10]))
    hidden_neurons_dense = int(np.interp(hidden_neurons_dense, [0, 255], [1, 10]))
    lr = round(np.interp(lr, [0, 255], [0.0001, 0.1]), 5)
    batch_size = int(np.interp(batch_size, [0, 255], [150, 3000]))

    output_channels = basis_func(output_channels, num_layers_conv)
    kernel_sizes = basis_func(kernel_sizes, num_layers_conv)
    stride_sizes = basis_func(stride_sizes, num_layers_conv)
    padding_sizes = basis_func(padding_sizes, num_layers_conv)
    hidden_neurons_dense = basis_func(hidden_neurons_dense, num_layers_conv)
    hidden_neurons_dense_arr = np.flip(np.array(hidden_neurons_dense))
    hidden_neurons_dense = list(hidden_neurons_dense_arr)
    hidden_neurons_dense.append(1)
    hidden_neurons_dense[-1] = 1

    hyperparameters = [seq_length, num_layers_conv, output_channels, kernel_sizes, stride_sizes, padding_sizes, hidden_size_lstm, num_layers_lstm, hidden_neurons_dense, lr, batch_size, n_epoch]
    print(f'hyperparameters: {hyperparameters}')
    return seq_length, num_layers_conv, output_channels, kernel_sizes, stride_sizes, padding_sizes, hidden_size_lstm, num_layers_lstm, hidden_neurons_dense, lr, batch_size, n_epoch, hyperparameters


