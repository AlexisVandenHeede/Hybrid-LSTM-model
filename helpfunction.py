import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import torch
from ParametricLSTMCNN import ParametricLSTMCNN
from bitstring import BitArray
from torch.utils.data import Dataset


def load_data_normalise_indv2(battery, model_type):
    debug = False
    """
    Load the data and normalize it.
    Return: normalized data, mean time, std time
    """
    data = {}
    means = []
    stds = []
    if model_type not in ['data', 'hybrid', 'data_padded', 'hybrid_padded']:
        print('Wrong model type, either data, hybrid, data_padded, or hybrid_padded')
        raise ValueError

    for i in battery:
        data_file = f"data/mod_volt[{i}].csv" if model_type == 'data' else f"data/{i}_TTD - with SOC.csv" if model_type == 'hybrid' else f"data/padded_data_mod_volt[{i}].csv" if model_type == 'data_padded' else f"data/padded_hybrid_mod_volt[{i}].csv"
        battery_data = pd.read_csv(data_file)
        time = battery_data['TTD']
        time_mean = time.mean(axis=0)
        time_std = time.std(axis=0)
        # normalized_battery_data = (battery_data - battery_data.min(axis=0)) / (battery_data.max(axis=0) - battery_data.min(axis=0))
        # normalized_battery_data = normalized_battery_data.astype('float16')  # Convert to float16 to save memory
        normalized_battery_data = (battery_data - battery_data.mean(axis=0)) / battery_data.std(axis=0)
        normalized_battery_data = normalized_battery_data.astype('float16')  # Convert to float16 to save memory
        data_name = f'data_bat_{i}'
        data[data_name] = normalized_battery_data
        means.append(time_mean)
        stds.append(time_std)
        # maxx.append(time_max)
        # minn.append(time_min)
 
    # data = pd.concat(data)
    if debug:
        print(data.columns)

    for i in battery:
        data_name = f'data_bat_{i}'
        data[data_name] = data[data_name].loc[:, ~data[data_name].columns.str.contains("^Unnamed")]  # Remove unnamed columns
        data[data_name] = data[data_name].loc[:, ~data[data_name].columns.str.contains("^Current")]  # Remove current columns

    # data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    # data = data.loc[:, ~data.columns.str.contains('^Current')]
    if debug:
        print(data.columns)

    return data, means, stds


def train_test_validation_split(X, y, test_size, cv_size):
    """
    The sklearn {train_test_split} function to split the dataset (and the labels) into
    train, test and cross-validation sets
    """
    X_train, X_test_cv, y_train, y_test_cv = train_test_split(
        X, y, test_size=test_size+cv_size, shuffle=False, random_state=0)

    test_size = test_size/(test_size+cv_size)

    X_cv, X_test, y_cv, y_test = train_test_split(
        X_test_cv, y_test_cv, test_size=test_size, shuffle=False, random_state=0)

    # return split data
    return X_train, y_train, X_test, y_test, X_cv, y_cv


def testing_func(X_test, y_test, model, criterion):
    """
    Return the rmse of the prediction from X_test compared to y_test
    """
    rmse_test = 0
    y_predict = model(X_test)
    rmse_test = np.sqrt(criterion(y_test, y_predict).item())
    return rmse_test


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


def train_batch_ind(model, train_dataloader, val_dataloader, n_epoch, lf, optimiser, verbose=False):
    """
    train model dataloaders, early stopper Class
    """
    epoch = []
    early_stopper = EarlyStopper(patience=10, min_delta=0.001)
    with torch.no_grad():
        train_loss_history = []
        val_loss_history = []

    for i in range(n_epoch):
        loss_v = 0
        loss = 0
        for l, (x, y) in enumerate(train_dataloader):
            model.train()
            target_train = model(x)
            loss_train = lf(target_train, y)
            loss += loss_train.item()
            epoch.append(i+1)
            optimiser.zero_grad()
            loss_train.backward()
            optimiser.step()
        train_loss = loss/len(train_dataloader)

        for k, (x, y) in enumerate(val_dataloader):
            model.eval()
            target_val = model(x)
            loss_val = lf(target_val, y)
            loss_v += loss_val.item()

        val_loss = loss_v/len(val_dataloader)
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        
        if verbose:
            print(f"Epoch {i+1}: train loss = {train_loss:.10f}, val loss = {val_loss:.10f}")
        # earlystopper
        if early_stopper.early_stop(val_loss):
            print("Early stopping")
            break
        if val_loss > 10:
            print('way too large val loss')
            break
        if i > 15:
            if train_loss > 0.4:
                print('no learning taking place')
                break
        if train_loss == 'nan':
            print('nan loss')
            break

    return model, train_loss_history, val_loss_history


def plot_loss(train_loss_history, val_loss_history):
    plt.plot(train_loss_history, label='train loss')
    plt.plot(val_loss_history, label='val loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()


def plot_predictions(model, X_test, y_test, ttd_mean, ttd_std, model_type, rmse):
    predictions = model(X_test)
    predictions = predictions.cpu() * ttd_std[0] + ttd_mean[0]
    y_test = y_test.cpu() * ttd_std[0] + ttd_mean[0]
    plt.plot(y_test.squeeze(), label='Actual')
    plt.plot(predictions.detach().squeeze(), label='Prediction')
    plt.xlabel('Time')
    plt.ylabel('TTD')
    plt.legend()
    plt.title(f'Predictions vs Actual for {rmse} model')
    plt.show()


def plot_average_predictionsv2(model, X_test, y_test, ttd_mean, ttd_std, model_type, rmse):
    debug = False
    df = pd.DataFrame()
    predictions = model(X_test)
    df['predictions'] = np.reshape(predictions.cpu().detach().numpy(), -1) * ttd_std + ttd_mean
    df.insert(1, 'y_test', np.reshape(y_test.cpu().detach().numpy(), -1) * ttd_std + ttd_mean)
    if debug: 
        print(df.describe())
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
    if debug:
        print(df.head())

    # Calculate scaled_cycle_index
    df['scaled_cycle_index'] = 0
    current_cycle_start = 0
    for length in cycle_lengths:
        scaled_indices = np.arange(length) / length
        df.loc[current_cycle_start:current_cycle_start + length - 1, 'scaled_cycle_index'] = scaled_indices
        current_cycle_start += length

    if debug:
        print(df.head())
        # plot scaled_cycle_index vs index
        plt.figure(figsize=(10, 6))
        plt.scatter(df.index, df['scaled_cycle_index'], label='scaled_cycle_index', marker='o')
        plt.xlabel('Index')
        plt.ylabel('Scaled Cycle Index')
        plt.legend()

        plt.show()

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


def plot_average_predictions(model, X_test, y_test, ttd_mean, ttd_std, model_type):
    debug = True
    df = pd.DataFrame()
    predictions = model(X_test)
    df['predictions'] = np.reshape(predictions.cpu().detach().numpy(), -1) * ttd_std + ttd_mean
    df.insert(1, 'y_test', np.reshape(y_test.cpu().detach().numpy(), -1) * ttd_std + ttd_mean)
    # print(df.describe())
    threshold = 2000
    cycle_starts = df['y_test'].diff().abs() > threshold
    if debug:
        predictions = model(X_test)
        predictions = predictions.cpu() * ttd_std + ttd_mean
        y_test = y_test.cpu() * ttd_std + ttd_mean
        plt.plot(y_test.squeeze(), label='Actual')
        plt.plot(predictions.detach().squeeze(), label='Prediction')
        print(type(cycle_starts))
        print(df[cycle_starts].index)
        # add cycle start as vertical line
        plt.vlines(x=df[cycle_starts].index, ymin=-200, ymax=3500, color='r', linestyles='dashed', label='Cycle start')
        plt.xlabel('Time')
        plt.ylabel('TTD')
        plt.legend()
        plt.show

    # linearly interpolate the data between each cycle start so that theyre all the same length

    cycle_mean = df.groupby(cycle_starts.cumsum())[["predictions", "y_test"]].mean()
    # print(cycle_mean.describe())
    plt.scatter(cycle_mean.index, cycle_mean['y_test'], label='Actual')
    plt.scatter(cycle_mean.index, cycle_mean['predictions'], label='Prediction')

    # plt.plot(y_test.squeeze(), label='Actual')
    # plt.plot(predictions.detach().squeeze(), label='Prediction')
    # plt.xlabel('Time')
    # plt.ylabel('TTD')
    plt.legend()
    plt.title(f'Average Predictions vs Actual for {model_type} model')
    plt.show()


class SeqDataset:
    def __init__(self, x_data, y_data, batch):
        self.x_data = x_data
        self.y_data = y_data
        self.batch = batch

    def __len__(self):
        return np.ceil((len(self.x_data) / self.batch)).astype('int')

    def __getitem__(self, idx):
        start_idx = idx * self.batch
        end_idx = start_idx + self.batch

        x = self.x_data[start_idx:end_idx]
        y = self.y_data[start_idx:end_idx]

        if end_idx > len(self.x_data):
            x = self.x_data[start_idx:]
            y = self.y_data[start_idx:]

        if x.shape[0] == 0:
            raise StopIteration

        return x, y


def eval_model(model, X_test, y_test, criterion):
    model.eval()
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
    if len(size_of_bat) == 1:
        x_tr = np.empty((len(X) - seq_length, seq_length, X.shape[1]))
        y_tr = np.empty((len(X) - seq_length, 1, 1))
        for i in range(seq_length, len(X)):
            x_tr[i - seq_length] = X.values[i - seq_length:i]
            y_tr[i - seq_length] = y.values[i]
    elif len(size_of_bat) == 2:
        x_tr_1 = np.empty((size_of_bat[0] - seq_length, seq_length, X.shape[1]))
        y_tr_1 = np.empty((size_of_bat[0] - seq_length, 1, 1))
        x_tr_2 = np.empty((size_of_bat[1] - seq_length, seq_length, X.shape[1]))
        y_tr_2 = np.empty((size_of_bat[1] - seq_length, 1, 1))
        for i in range(seq_length, size_of_bat[0]):
            x_tr_1[i - seq_length] = X.values[i - seq_length:i]
            y_tr_1[i - seq_length] = y.values[i]
        for i in range(seq_length, size_of_bat[1]):
            x_tr_2[i - seq_length] = X.values[i - seq_length:i]
            y_tr_2[i - seq_length] = y.values[i]
        x_tr = np.concatenate((x_tr_1, x_tr_2), axis=0)
        y_tr = np.concatenate((y_tr_1, y_tr_2), axis=0)

    x_tr = torch.tensor(x_tr)
    y_tr = torch.tensor(y_tr)
    print(x_tr.shape)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_tr = x_tr.to(device).float()
    y_tr = y_tr.to(device).float()
    return x_tr, y_tr


def kfold_ind(model_type, hyperparameters, battery, plot=False, strict=True):
    print(f'model type = {model_type}')
    k_fold_rmse = []
    k_fold_raw_test = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    for i in range(4):
        torch.cuda.empty_cache()
        battery_temp = battery.copy()
        test_battery = [battery[i]]
        # print(f'test battery is {test_battery}')
        battery_temp.remove(test_battery[0])
        if i == 3:
            validation_battery = [battery[0]]
        else:
            validation_battery = [battery[i+1]]
        battery_temp.remove(validation_battery[0])
        # print(f'validation battery is {validation_battery}')
        train_battery = battery_temp
        print(f'train batteries are {train_battery}')
        normalised_data_train, means_train, stds_train = load_data_normalise_indv2(train_battery, model_type)
        normalised_data_test, means_test, stds_test = load_data_normalise_indv2(test_battery, model_type)
        normalised_data_validation, means_val, stds_val = load_data_normalise_indv2(validation_battery, model_type)
        seq_length = hyperparameters[0]

        X_train_1, y_train_1, X_train_2, y_train_2 = seq_split(train_battery, normalised_data_train, means_train, stds_train, seq_length, model_type)
        X_test, y_test = seq_split(test_battery, normalised_data_test, means_test, stds_test, seq_length, model_type)
        X_validation, y_validation = seq_split(validation_battery, normalised_data_validation, means_val, stds_val, seq_length, model_type)
        # print(f'X_train_1 shape = {X_train_1.shape}', f'y_train_1 shape = {y_train_1.shape}')
        # print(f'validation shape = {X_validation.shape}, {y_validation.shape}')
        model = ParametricLSTMCNN(num_layers_conv=hyperparameters[1], output_channels=hyperparameters[2], kernel_sizes=hyperparameters[3], stride_sizes=hyperparameters[4], padding_sizes=hyperparameters[5], hidden_size_lstm=hyperparameters[6], num_layers_lstm=hyperparameters[7], hidden_neurons_dense=hyperparameters[8], seq=seq_length, inputlstm=X_train_1.shape[2])
        model.weights_init()
        lf = torch.nn.MSELoss()
        opimiser = torch.optim.Adam(model.parameters(), lr=hyperparameters[9])
        model.to(device)
        train_dataset = SeqDataset(x_data=X_train_1, y_data=y_train_1, batch=hyperparameters[10])
        validation_dataset = SeqDataset(x_data=X_validation, y_data=y_validation, batch=hyperparameters[10])
        model, train_loss_history, val_loss_history = train_batch_ind(model, train_dataset, validation_dataset, n_epoch=hyperparameters[11], lf=lf, optimiser=opimiser, verbose=True)
        
        rmse_test, raw_test = eval_model(model, X_test, y_test, lf)
        print(f'rmse_test = {rmse_test}')
        if plot and rmse_test < 0.20:
            print(f'i will now plot a figure')
            # plot_loss(train_loss_history, val_loss_history)
            # plot_predictions(model, X_test, y_test, means_test, stds_test, model_type, rmse_test)
            plot_average_predictionsv2(model, X_test, y_test, means_test, stds_test, model_type, rmse_test)

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
    if rmse_test < 0.27:
        # save model parameters
        print(f'saving model with rmse_test = {rmse_test} and raw_err = {raw_test}')
        print(f'hyperparameters are: {hyperparameters}')
        torch.save(model.state_dict(), f'best_model_rmse_{rmse_test}.pt')

    print(f'average rmse_test = {rmse_test} and raw_err = {raw_test}')
    return rmse_test


def seq_split(battery, normalised_data, mean, std, seq_length, model_type):
    # device = torch.device('cpu')

    for i in range(len(battery)):
        data_name = f'data_bat_{battery[i]}'
        if model_type == 'data_padded' or model_type == 'data':
            X = normalised_data[data_name].drop(['TTD', 'Time', 'Start_time'], axis=1)
        elif model_type == 'hybrid_padded':
            X = normalised_data[data_name].drop(['TTD', 'Time', 'Start_time', 'Instance', 'Voltage_measured'], axis=1)
        y = normalised_data[data_name]['TTD']
        
        y_not_norm = pd.Series(y * std[i] + mean[i])
        indx = np.array((y_not_norm.index[y_not_norm == y_not_norm.min()]).to_list())
        
        indx_diff = np.diff(indx)
        # print(f'indx_diff = {indx_diff}')
        # print(np.sum(indx_diff) - (len(indx)-1) * seq_length + 2*len(indx)-1)

        xtr_batch = np.zeros(((np.sum(indx_diff) - (len(indx)-1) * seq_length + 2*len(indx)-1), seq_length, X.shape[1]))
        ytr_batch = np.zeros(((np.sum(indx_diff) - (len(indx)-1) * seq_length + 2*len(indx)-1), 1, 1))

        counter = 0
        for j in range(len(indx) - 1):
            for k in range(indx_diff[j] - seq_length + 2):
                xtr_batch[counter] = X.values[counter:seq_length+counter]
                ytr_batch[counter] = y.values[counter+seq_length]
                counter += 1
        
        # print(f'xtr_batch shape = {xtr_batch.shape}, ytr_batch shape = {ytr_batch.shape}')
        xtr_batch = torch.tensor(xtr_batch)
        ytr_batch = torch.tensor(ytr_batch)
        # plt.plot(ytr_batch.detach().squeeze().cpu().numpy(), label='ytr_batch')
        # plt.show()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if i == 0:
            x_tr_1 = xtr_batch.to(device).float()
            y_tr_1 = ytr_batch.to(device).float()
        else:
            x_tr_2 = xtr_batch.to(device).float()
            y_tr_2 = ytr_batch.to(device).float()

    if len(battery) == 1:
        return x_tr_1, y_tr_1
    elif len(battery) == 2:
        return x_tr_1, y_tr_1, x_tr_2, y_tr_2
    

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
    seq_length = int(np.interp(seq_length, [0, 255], [50, 120]))
    num_layers_conv = int(np.interp(num_layers_conv, [0, 255], [1, 5]))
    output_channels = int(np.interp(output_channels, [0, 255], [1, 10]))
    kernel_sizes = int(np.interp(kernel_sizes, [0, 255], [1, 10]))
    stride_sizes = int(np.interp(stride_sizes, [0, 255], [1, 10]))
    padding_sizes = int(np.interp(padding_sizes, [0, 255], [1, 10]))
    hidden_size_lstm = int(np.interp(hidden_size_lstm, [0, 255], [1, 10]))
    num_layers_lstm = int(np.interp(num_layers_lstm, [0, 255], [1, 4]))
    hidden_neurons_dense = int(np.interp(hidden_neurons_dense, [0, 255], [1, 10]))
    lr = round(np.interp(lr, [0, 255], [0.0001, 0.1]), 5)
    batch_size = int(np.interp(batch_size, [0, 255], [150, 500]))

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


def extrapolate(bat):
    data_file = f'data/{bat}_TTD1.csv'
    battery_data = pd.read_csv(data_file)
    time = battery_data['TTD']
    voltage = battery_data['Voltage_measured']
    indx = time.index[time == time.min()].tolist()
    indx = [0] + indx
    # print(indx)
    voltage_ext = []
    time_ext = []
    # print(indx)

    for i in range(len(indx)-1):
        for k in (range(indx[i], indx[i+1])):
            # print(voltage[k-1], voltage[k])
            if k == indx[i]:
                voltage_ext.append(voltage[k])
                time_ext.append(time[k])
            elif voltage[k-1] > voltage[k]:
                voltage_ext.append(voltage[k])
                time_ext.append(time[k])
            else:
                # print(f'k = {k} and i = {i}')
                fit = np.polyfit(time_ext, voltage_ext, 9)
                fit_fn = np.poly1d(fit)
                voltage_ext.append(fit_fn(time[k]))
                time_ext.append(time[k])
                # print(voltage[k-1], voltage[k], k, fit_fn(time[k]))
        # plt.plot(time_ext, voltage_ext)
        # plt.plot(time_ext, voltage[indx[i]:indx[i+1]])
        # plt.show()
    voltage_ext.append(voltage[indx[-1]])
    time_ext.append(time[indx[-1]])
    battery_data['Voltage_measured'] = voltage_ext

    data_file = f'data/mod_volt[{bat}].csv'
    battery_data.to_csv(data_file, index=False)
    return print('done')

# extrapolate('B0005')
            
