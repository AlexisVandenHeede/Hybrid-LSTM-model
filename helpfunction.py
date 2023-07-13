import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def load_data_normalise(battery, model_type):
    """
    Load the data and normalise it
    return: normalised data, mean time, std time
    """
    data = []
    if model_type == 'data':
        for i in battery:
            data.append(pd.read_csv("data/" + i + "_TTD.csv"))
    elif model_typev == 'hybrid':
        for i in battery:
            data.append(pd.read_csv("data/" + i + "_TTD - with SOC.csv"))
    else:
        print('wrong model type, either data or hybrid')
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

def data_split(normalised_data, test_size, cv_size):
    """ Split data into X Y  train, test and validation sets"""
    y = normalised_data['TTD']
    X = normalised_data.drop(['TTD', 'Time'], axis=1)
    X_train, y_train, X_test, y_test, X_cv, y_cv  = train_test__validation_split(X, y, test_size, cv_size)
    return X_train, y_train, X_test, y_test, X_cv, y_cv

def testing_func(X_test, y_test):
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
    
    def early_stop(self, validaiton_loss):
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
    basis = (np.arange(hidden_layers, dtype=int))*scaling_factor
    if hidden_layers == 1:
        basis[0] = 1
    basis_function = []
    for i in range(hidden_layers):
        if basis[i] == 0:
            basis[i] = 1
        basis_function.append(basis[i])
    return basis_function

def trainbatch(model, train_dataloader, val_dataloader, n_epochs, lf, optimiser, verbose):
    """
    train model dataloaders, early stopper Class
    """
    epoch = []
    early_stopper = EarlyStopper(patience=10, min_delta=0.0001)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    with torch.no_grad():
        train_loss_history = []
        val_loss_history = []

    for i in range(n_epoch):
        loss_v = 0
        loss = 0
        for l, (x, y) in enumerate(train_dataloader):
            target_train = model(x)
            loss_train = lf(target, y)
            loss += loss_train.item()
            epoch.append(i+1)
            optimiser.zero_grad()
            loss_train.backward()
            optimiser.step()
        
        for k, (x, y) in enumerate(val_dataloader):
            target_val = model(x)
            loss_val = lf(target_val, y)
            loss_v += loss_val.item()
        
        train_loss = loss/len(train_dataloader)
        val_loss = loss_v/len(val_dataloader)
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        if verbose:
            val_loss = np.array(val_loss_history)
            print(f"Epoch {i+1}: train loss = {train_loss:.10f}, val loss = {val_loss:.10f}")
        # earlystopper
        if early_stopper.early_stop(val_loss):
            print("Early stopping")
            break
        return model, train_loss_history, val_loss_history

def plot_loss(train_loss_history, val_loss_history):
    plt.plot(train_loss_history, label = 'train loss')
    plt.plot(val_loss_history, label = 'val loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

class SeqDataset:
    def __init__(self, x_data, y_data, seq_len, batch):
        self.x_data = x_data
        self.y_data = y_data
        self.seq_len = seq_len
        self.batch = batch

    def __len__(self):
        return np.ceil((len(self.x_data) / self.batch))

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