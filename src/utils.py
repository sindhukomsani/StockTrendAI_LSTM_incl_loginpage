import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

def load_close_series(csv_path, date_col='Date', close_col='Close'):
    """
    Loads CSV and returns DataFrame with Date and Close sorted ascending.
    """
    df = pd.read_csv(csv_path, parse_dates=[date_col])
    df = df.sort_values(by=date_col)
    df = df[[date_col, close_col]].rename(columns={date_col: 'date', close_col: 'close'})
    df = df.reset_index(drop=True)
    return df

def scale_series(series, scaler=None):
    """
    Scales close price series using MinMaxScaler.
    """
    arr = np.array(series).reshape(-1, 1).astype('float32')
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(arr)
    else:
        scaled = scaler.transform(arr)
    return scaled, scaler

def make_sequences(data, n_steps=10):
    """
    Converts data into sequences for LSTM input.
    """
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps])
    return np.array(X), np.array(y)
