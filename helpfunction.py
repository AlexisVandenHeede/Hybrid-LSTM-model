import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import torch
import math
from custom_types import Model, Battery


def load_data_normalise(battery_list: list[Battery], model_type: Model) -> tuple[pd.DataFrame, float, float]:
    """
    Load the data and normalise it
    return: normalised data, mean time, std time
    """
    data_list: list[pd.DataFrame] = []
    if model_type == 'data':
        for i in battery_list:
            data_list.append(pd.read_csv("data/" + i + "_TTD1.csv"))
    elif model_type == 'hybrid':
        for i in battery_list:
            data_list.append(pd.read_csv("data/" + i + "_TTD - with SOC.csv"))
    else:
        raise NameError('model type must be either data or hybrid')
    data: pd.DataFrame = pd.concat(data_list)
    time = data['Time']
    time_mean = time.mean(axis=0)
    time_std = time.std(axis=0)
    normalised_data = (data - data.mean(axis=0)) / data.std(axis=0)
    return normalised_data, time_mean, time_std


def train_test_validation_split(
        X: pd.DataFrame, y: pd.Series, test_size: float, cv_size: float
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
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


def data_split(
        normalised_data: pd.DataFrame, test_size: float, cv_size: float, seq_length: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """ Split data into X Y  train, test and validation sets"""
    y = normalised_data['TTD']
    X = normalised_data.drop(['TTD', 'Time'], axis=1)
    X_train, y_train, X_test, y_test, X_cv, y_cv = train_test_validation_split(X, y, test_size, cv_size)

    x_tr_list = []
    y_tr_list = []
    for i in range(seq_length, len(X_train)):
        x_tr_list.append(X_train.values[i-seq_length:i])
        y_tr_list.append(y_train.values[i])

    x_tr = torch.tensor(np.array(x_tr_list))
    y_tr = torch.tensor(y_tr_list).unsqueeze(1).unsqueeze(2)

    x_v_list = []
    y_v_list = []
    for i in range(seq_length, len(X_cv)):
        x_v_list.append(X_cv.values[i-seq_length:i])
        y_v_list.append(y_cv.values[i])

    x_v = torch.tensor(np.array(x_v_list))
    y_v = torch.tensor(y_v_list).unsqueeze(1).unsqueeze(2)

    x_t_list = []
    y_t_list = []
    for i in range(seq_length, len(X_test)):
        x_t_list.append(X_test.values[i-seq_length:i])
        y_t_list.append(y_test.values[i])

    x_t = torch.tensor(np.array(x_t_list))
    y_t = torch.tensor(y_t_list).unsqueeze(1).unsqueeze(2)

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
    # NOTE: type hints probably not necessary here, this is fairly generic
    rmse_test = 0
    y_predict = model(X_test)
    rmse_test = np.sqrt(criterion(y_test, y_predict).item())
    return rmse_test


class EarlyStopper:
    def __init__(self, patience: int, min_delta: float):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss: float) -> bool:
        """Implement the early stopping criterion.
        The function has to return 'True' if the current validation loss (in the arguments) has increased
        with respect to the minimum value of more than 'min_delta' and for more than 'patience' steps.
        Otherwise the function returns 'False'."""
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            return False
        else:
            if (validation_loss - self.min_validation_loss) > self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    return True
                else:
                    return False
            else:
                return False


def basis_func(scaling_factor, hidden_layers):
    """ Rescale hyperparameter per layer using basis function, now just np.arange"""
    # TODO: type hint once I (Jeremy) understand what this is doing
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
    early_stopper = EarlyStopper(patience=10, min_delta=0.0001)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    with torch.no_grad():
        train_loss_history = []
        val_loss_history = []

    for i in range(n_epoch):
        loss_v = 0
        loss = 0
        for (x, y) in train_dataloader:
            model.train()
            target_train = model(x)
            loss_train = lf(target_train, y)
            loss += loss_train.item()
            epoch.append(i+1)
            optimiser.zero_grad()
            loss_train.backward()
            optimiser.step()

        for (x, y) in val_dataloader:
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


class SeqDataset:
    # TODO: check why this doesn't just use a dataloader as used in eg.
    # https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#iterating-through-the-dataset
    def __init__(self, x_data: torch.Tensor, y_data: torch.Tensor, seq_len: int, batch: int):
        self.x_data = x_data
        self.y_data = y_data
        self.seq_len = seq_len
        self.batch = batch

    def __len__(self):
        return math.ceil((len(self.x_data) / self.batch))

    def __getitem__(self, idx: int):
        start_idx = idx * self.batch
        end_idx = start_idx + self.batch

        if end_idx > len(self.x_data):
            x = self.x_data[start_idx:]
            y = self.y_data[start_idx:]
        else:
            x = self.x_data[start_idx:end_idx]
            y = self.y_data[start_idx:end_idx]

        if x.shape[0] == 0:
            raise StopIteration

        return x, y


def eval_model(model: torch.nn.Module, X_test: torch.Tensor, y_test: torch.Tensor, criterion) -> float:
    """
    WIP
    """
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        rmse: float = np.sqrt(criterion(y_test, y_pred).item())
    return rmse
