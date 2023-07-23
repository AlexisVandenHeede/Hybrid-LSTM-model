import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import torch
from ParametricLSTMCNN import ParametricLSTMCNN
from bitstring import BitArray


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
    data = pd.concat(data)
    TTD = data['TTD']
    index_jumps = TTD.where(TTD == 0, 1)
    new_cycle = TTD.where(TTD.diff() < 0, 1)
    new_cycle[0] = 1
    new_cycle[new_cycle != 1] = 0
    new_cycle *= n
    index_jumps = index_jumps.replace({0: 1, 1: 0}) * n
    # print(f'index_jumps = {index_jumps}')
    # print(data)
    new_data = data.index.repeat(index_jumps)
    # print(new_data)
    new_data = pd.concat([data, data.iloc[new_data]])
    new_data_1 = data.index.repeat(new_cycle)
    new_data = pd.concat([new_data, data.iloc[new_data_1]])
    # print(new_data)
    new_data.sort_index(inplace=True)
    new_data.reset_index().to_csv(f'data/padded_data_{model_type}_{battery}.csv')
    return print('padded data saved')


def load_data_normalise(battery, model_type):
    """
    Load the data and normalise it
    return: normalised data, mean time, std time
    """
    data = []
    if model_type == 'data':
        for i in battery:
            data.append(pd.read_csv("data/" + i + "_TTD1.csv"))
    elif model_type == 'hybrid':
        for i in battery:
            data.append(pd.read_csv("data/" + i + "_TTD - with SOC.csv"))
    elif model_type == 'data_padded':
        for i in battery:
            data.append(pd.read_csv(f"data/padded_data_data_[{i}].csv"))
    elif model_type == 'hybrid_padded':
        for i in battery:
            data.append(pd.read_csv(f"data/padded_data_hybrid_w_ecm[{i}].csv"))
    else:
        print('wrong model type, either data or hybrid or data_padded or hybrid_padded')
        raise NameError
    data = pd.concat(data)
    time = data['Time']
    time_mean = time.mean(axis=0)
    time_std = time.std(axis=0)
    normalised_data = (data - data.mean(axis=0)) / data.std(axis=0)
    return normalised_data, time_mean, time_std


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


def data_split(normalised_data, test_size, cv_size, seq_length):
    """ Split data into X Y  train, test and validation sets"""
    y = normalised_data['TTD']
    X = normalised_data.drop(['TTD', 'Time'], axis=1)
    X_train, y_train, X_test, y_test, X_cv, y_cv = train_test_validation_split(X, y, test_size, cv_size)
    x_tr = []
    y_tr = []
    # this for loop is very inefficient, as it fills ram
    # for i in range(seq_length, len(X_train)):
    #     print('2')
    #     x_tr.append(X_train.values[i-seq_length:i])
    #     y_tr.append(y_train.values[i])
    x_tr = X_train.values[:]
    y_tr = y_train.values[seq_length:]
    x_tr = np.array([x_tr[i-seq_length:i] for i in range(seq_length, len(x_tr))])
    x_tr = torch.tensor(np.array(x_tr))
    y_tr = torch.tensor(y_tr).unsqueeze(1).unsqueeze(2)
    x_v = []
    y_v = []
    # for i in range(seq_length, len(X_cv)):
    #     x_v.append(X_cv.values[i-seq_length:i])
    #     y_v.append(y_cv.values[i])
    x_v = X_cv.values[:]
    y_v = y_cv.values[seq_length:]
    x_v = np.array([x_v[i-seq_length:i] for i in range(seq_length, len(x_v))])
    x_v = torch.tensor(np.array(x_v))
    y_v = torch.tensor(y_v).unsqueeze(1).unsqueeze(2)
    x_t = []
    y_t = []
    # for i in range(seq_length, len(X_test)):
    #     x_t.append(X_test.values[i-seq_length:i])
    #     y_t.append(y_test.values[i])
    x_t = X_test.values[:]
    y_t = y_test.values[seq_length:]
    x_t = np.array([x_t[i-seq_length:i] for i in range(seq_length, len(x_t))])
    x_t = torch.tensor(np.array(x_t))
    y_t = torch.tensor(y_t).unsqueeze(1).unsqueeze(2)

    if torch.cuda.is_available():
        print('Running on GPU')
        X_train = x_tr.to('cuda').float()
        y_train = y_tr.to('cuda').float()
        X_test = x_t.to('cuda').float()
        y_test = y_t.to('cuda').float()
        X_cv = x_v.to('cuda').float()
        y_cv = y_v.to('cuda').float()
        print("X_train and y_train are on GPU: ", X_train.is_cuda, y_train.is_cuda)
        print("X_test and y_test are on GPU: ", X_test.is_cuda, y_test.is_cuda)
        print("X_cv and y_cv are on GPU: ", X_cv.is_cuda, y_cv.is_cuda)
        print(f"size of X_train: {X_train.size()} and y_train: {y_train.size()}")
    else:
        X_train = x_tr.clone().detach().float()
        y_train = y_tr.clone().detach().float()
        X_test = x_t.clone().detach().float()
        y_test = y_t.clone().detach().float()
        X_cv = x_v.clone().detach().float()
        y_cv = y_v.clone().detach().float()

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


def train_batch(model, train_dataloader, val_dataloader, n_epoch, lf, optimiser, verbose):
    """
    train model dataloaders, early stopper Class
    """
    epoch = []
    early_stopper = EarlyStopper(patience=5, min_delta=0.00001)
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

        for k, (x, y) in enumerate(val_dataloader):
            model.eval()
            target_val = model(x)
            loss_val = lf(target_val, y)
            loss_v += loss_val.item()

        train_loss = loss/len(train_dataloader)
        val_loss = loss_v/len(val_dataloader)
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        if verbose:
            print(f"Epoch {i+1}: train loss = {train_loss:.10f}, val loss = {val_loss:.10f}")
        # earlystopper
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


def plot_predictions(model, X_test, y_test, ttd_mean, ttd_std, model_type):
    predictions = model(X_test)
    predictions = predictions.cpu() * ttd_std + ttd_mean
    y_test = y_test.cpu() * ttd_std + ttd_mean
    plt.plot(y_test.squeeze(), label='Actual')
    plt.plot(predictions.detach().squeeze(), label='Prediction')
    plt.xlabel('Time')
    plt.ylabel('TTD')
    plt.legend()
    plt.title(f'Predictions vs Actual for {model_type} model')
    plt.show()


class SeqDataset:
    def __init__(self, x_data, y_data, seq_len, batch):
        self.x_data = x_data
        self.y_data = y_data
        self.seq_len = seq_len
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
    """
    WIP
    """
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        rmse = np.sqrt(criterion(y_test, y_pred).item())
    print(f'rmse_test = {rmse}')
    return rmse


def k_fold(model_type, hyperparameters, battery, verbose, strict):
    k_fold_rmse = []
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
        print(f'train battery is {train_battery}')
        normalised_data_train, time_mean_train, time_std_train = load_data_normalise(train_battery, model_type)
        normalised_data_test, time_mean_test, time_std_test = load_data_normalise(test_battery, model_type)
        normalised_data_validation, time_mean_validation, time_std_validation = load_data_normalise(validation_battery, model_type)
        seq_length = hyperparameters[0]
        X_train, y_train = k_fold_data(normalised_data_train, seq_length)
        X_test, y_test = k_fold_data(normalised_data_test, seq_length)
        X_validation, y_validation = k_fold_data(normalised_data_validation, seq_length)
        model = ParametricLSTMCNN(hyperparameters[1], hyperparameters[2], hyperparameters[3], hyperparameters[4], hyperparameters[5], hyperparameters[6], hyperparameters[7], hyperparameters[8], hyperparameters[0], X_train.shape[2])
        lf = torch.nn.MSELoss()
        opimiser = torch.optim.Adam(model.parameters(), lr=hyperparameters[9])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        train_dataset = SeqDataset(x_data=X_train, y_data=y_train, seq_len=seq_length, batch=hyperparameters[10])
        validation_dataset = SeqDataset(x_data=X_validation, y_data=y_validation, seq_len=seq_length, batch=hyperparameters[10])
        model, train_loss_history, val_loss_history = train_batch(model, train_dataset, validation_dataset, n_epoch=hyperparameters[11], lf=lf, optimiser=opimiser, verbose=True)
        rmse_test = eval_model(model, X_test, y_test, lf)
        print(f'rmse_test = {rmse_test}')
        if verbose:
            plot_loss(train_loss_history, val_loss_history)
            plot_predictions(model, X_test, y_test, model_type)
        k_fold_rmse.append(rmse_test)
        if strict:
            if sum(k_fold_rmse) > (i+1):
                print(f'sum = {sum(k_fold_rmse)}')
                print(f'rmse too high')
                k_fold_rmse = 100
                break
    rmse_test = np.mean(k_fold_rmse)
    print(f'average rmse_test = {rmse_test}')
    return rmse_test


def k_fold_data(normalised_data, seq_length):
    y = normalised_data['TTD']
    X = normalised_data.drop(['TTD', 'Time'], axis=1)
    x_tr = X.values[:]
    y_tr = y.values[seq_length:]
    x_tr = np.array([x_tr[i-seq_length:i] for i in range(seq_length, len(x_tr))])
    x_tr = torch.tensor(np.array(x_tr))
    y_tr = torch.tensor(y_tr).unsqueeze(1).unsqueeze(2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_tr = x_tr.to(device).float()
    y_tr = y_tr.to(device).float()
    return x_tr, y_tr


def kfold_ind(model_type, hyperparameters, battery, plot=False, strict=False):
    k_fold_rmse = []
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
        train_battery_1 = [battery_temp[0]]
        train_battery_2 = [battery_temp[1]]
        print(f'train batteries are {train_battery_1} and {train_battery_2}')
        normalised_data_train_1, time_mean_train, time_std_train = load_data_normalise(train_battery_1, model_type)
        normalised_data_train_2, time_mean_train, time_std_train = load_data_normalise(train_battery_2, model_type)
        normalised_data_test, time_mean_test, time_std_test = load_data_normalise(test_battery, model_type)
        normalised_data_validation, time_mean_validation, time_std_validation = load_data_normalise(validation_battery, model_type)
        seq_length = hyperparameters[0]
        X_train_1, y_train_1 = k_fold_data(normalised_data_train_1, seq_length)
        X_train_2, y_train_2 = k_fold_data(normalised_data_train_2, seq_length)
        X_test, y_test = k_fold_data(normalised_data_test, seq_length)
        X_validation, y_validation = k_fold_data(normalised_data_validation, seq_length)
        model = ParametricLSTMCNN(hyperparameters[1], hyperparameters[2], hyperparameters[3], hyperparameters[4], hyperparameters[5], hyperparameters[6], hyperparameters[7], hyperparameters[8], hyperparameters[0], X_train_1.shape[2])
        lf = torch.nn.MSELoss()
        opimiser = torch.optim.Adam(model.parameters(), lr=hyperparameters[9])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        train_dataset_1 = SeqDataset(x_data=X_train_1, y_data=y_train_1, seq_len=seq_length, batch=hyperparameters[10])
        train_dataset_2 = SeqDataset(x_data=X_train_2, y_data=y_train_2, seq_len=seq_length, batch=hyperparameters[10])
        validation_dataset = SeqDataset(x_data=X_validation, y_data=y_validation, seq_len=seq_length, batch=hyperparameters[10])
        model, train_loss_history, val_loss_history = train_batch(model, train_dataset_1, validation_dataset, n_epoch=hyperparameters[11], lf=lf, optimiser=opimiser, verbose=True)
        model, train_loss_history, val_loss_history = train_batch(model, train_dataset_2, validation_dataset, n_epoch=hyperparameters[11], lf=lf, optimiser=opimiser, verbose=True)
        rmse_test = eval_model(model, X_test, y_test, lf)
        print(f'rmse_test = {rmse_test}')
        if plot:
            plot_loss(train_loss_history, val_loss_history)
            plot_predictions(model, X_test, y_test, time_mean_test, time_std_test, model_type)
        k_fold_rmse.append(rmse_test)
        if strict:
            if sum(k_fold_rmse) > (i+1):
                print(f'sum = {sum(k_fold_rmse)}')
                print(f'rmse too high')
                k_fold_rmse = 100
                break
    rmse_test = np.mean(k_fold_rmse)
    print(f'average rmse_test = {rmse_test}')
    return rmse_test


def add_ecm_data(battery_num):
    df_ecm = pd.read_csv(f'ECM/ECM_data_{battery_num}.csv')
    df_padded = pd.read_csv(f'data/padded_data_data_[{battery_num}].csv')
    df_padded['Instance'] = [i for i in range(0, len(df_padded))]
    soc = np.ones(len(df_padded))
    clean = np.ones(len(df_padded))
    for i in df_ecm['Instance']:
        for j in df_padded['Instance']:
            if i == j+1:
                soc[j] = df_ecm.loc[df_ecm['Instance'] == i, 'SOC'].iloc[0]
                clean[j] = df_ecm.loc[df_ecm['Instance'] == i, 'Vt_est'].iloc[0]
                break
    for i in range(len(soc)):
        if soc[i] == 1 and df_padded['TTD'][i] < 100:
            soc[i] = soc[i-1]
            clean[i] = df_padded['Voltage_measured'][i]
        elif soc[i] == 1 and df_padded['TTD'][i] > 1000:
            soc[i] = 0.85
            clean[i] = df_padded['Voltage_measured'][i]
    df_padded['SOC'] = soc
    df_padded['Vt_est'] = clean
    df_padded.to_csv(f'data/padded_data_hybrid_w_ecm[{battery_num}].csv')
    return print('ECM data added')


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
    return hyperparameters
