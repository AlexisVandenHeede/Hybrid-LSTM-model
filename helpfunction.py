import pandas as pd


def load_data_normalise(battery, model):
    """
    Load the data and normalise it
    return: normalised data, mean time, std time
    """
    data = []
    if model == 'data':
        for i in battery:
            data.append(pd.read_csv("data/" + i + "_TTD.csv"))
    elif model == 'hybrid':
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
