# archived help functions
import pandas as pd
import numpy as np
import torch 
from helpfunction import SeqDataset,EarlyStopper, eval_model, plot_loss, plot_predictions, train_test_validation_split
from ParametricLSTMCNN import ParametricLSTMCNN


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


def remove_voltage(battery_num):
    df_padded = pd.read_csv(f'data/padded_data_data_[{battery_num}].csv')
    voltage = df_padded['Voltage_measured']
    ttd = df_padded['TTD']
    for i in range(len(voltage)-1):
        if voltage[i+1] > voltage[i] and ttd[i] < 700:
            voltage[i+1] = voltage[i]
    
    df_padded['Voltage_measured'] = voltage
    df_padded.to_csv(f'data/padded_data_mod_volt[{battery_num}].csv')
    return print('voltage removed')


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

    return model, train_loss_history, val_loss_history


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
            if np.mean(k_fold_rmse) > 1:
                print(f'average = {np.mean(k_fold_rmse)}')
                print(f'rmse too high')
                k_fold_rmse = 100
                break
    rmse_test = np.mean(k_fold_rmse)
    print(f'average rmse_test = {rmse_test}')
    return rmse_test


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


def data_split(normalised_data, test_size, cv_size, seq_length, model_type):
    """ Split data into X Y  train, test and validation sets"""
    y = normalised_data['TTD']
    if model_type == 'data_padded' or model_type == 'data':
        X = normalised_data.drop(['TTD', 'Time', 'Start_time', 'Unnamed: 0', 'Unnamed: 0.1'], axis=1)
    elif model_type == 'hybrid_padded':
        X = normalised_data.drop(['TTD', 'Time', 'Start_time', 'Instance', 'Voltage_measured', 'Unnamed: 0.1', 'Unnamed: 0'], axis=1)
    X_train, y_train, X_test, y_test, X_cv, y_cv = train_test_validation_split(X, y, test_size, cv_size)
    x_tr = []
    y_tr = []
    x_tr = X_train.values[:]
    y_tr = y_train.values[seq_length:]
    x_tr = np.array([x_tr[i-seq_length:i] for i in range(seq_length, len(x_tr))])
    x_tr = torch.tensor(np.array(x_tr))
    y_tr = torch.tensor(y_tr).unsqueeze(1).unsqueeze(2)

    x_v = []
    y_v = []
    x_v = X_cv.values[:]
    y_v = y_cv.values[seq_length:]
    x_v = np.array([x_v[i-seq_length:i] for i in range(seq_length, len(x_v))])
    x_v = torch.tensor(np.array(x_v))
    y_v = torch.tensor(y_v).unsqueeze(1).unsqueeze(2)

    x_t = []
    y_t = []
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


def k_fold_data(normalised_data, seq_length, model_type, size_of_bat):
    if model_type == 'data_padded' or model_type == 'data':
        X = normalised_data.drop(['TTD', 'Time', 'Start_time'], axis=1)
    elif model_type == 'hybrid_padded':
        X = normalised_data.drop(['TTD', 'Time', 'Start_time', 'Instance', 'Voltage_measured',], axis=1)
    y = normalised_data['TTD']
    # print(f'shape of x and y is {X.shape}, {y.shape}')
    x_tr = []
    y_tr = []
    for i in range(len(size_of_bat)):
        if len(size_of_bat) == 1:
            x_tr = []
            y_tr = []
            for i in range(seq_length, len(X)):
                x_tr.append(X.values[i-seq_length:i])
                y_tr.append(y.values[i])
            x_tr = np.array(x_tr)
            y_tr = np.array(y_tr)
        if len(size_of_bat) == 2:
            x_tr_1 = []
            y_tr_1 = []
            x_tr_2 = []
            y_tr_2 = []
            for i in range(seq_length, size_of_bat[0]):
                x_tr_1.append(X.values[i-seq_length:i])
                y_tr_1.append(y.values[i])
            for i in range(seq_length, size_of_bat[1]):
                x_tr_2.append(X.values[i-seq_length:i])
                y_tr_2.append(y.values[i])
            x_tr = np.concatenate((np.array(x_tr_1), np.array(x_tr_2)), axis=0)
            y_tr = np.concatenate((np.array(y_tr_1), np.array(y_tr_2)), axis=0)
    
    x_tr = torch.tensor((x_tr))
    y_tr = torch.tensor((y_tr)).unsqueeze(1).unsqueeze(2)
    # print(f'shape of x_tr is {x_tr.shape}, shape of y_tr is {y_tr.shape}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_tr = x_tr.to(device).float()
    y_tr = y_tr.to(device).float()
    return x_tr, y_tr