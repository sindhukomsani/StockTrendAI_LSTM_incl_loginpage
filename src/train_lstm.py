# src/train_lstm.py
import os
import argparse
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
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

def make_sequences(data, n_steps=10):
    """
    data: numpy array (scaled) shape=(N,1)
    returns X (num_samples, n_steps, 1) and y (num_samples,)
    """
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps])
    X = np.array(X)
    y = np.array(y)
    return X, y

def train(csv_path, model_path, scaler_path, n_steps=10, epochs=10, batch_size=16):
    print("🔹 Loading data from:", csv_path)
    df = load_close_series(csv_path)
    print("✅ CSV loaded successfully! Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df.head())

    # Scale values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df['close'].values.reshape(-1, 1))

    # Create sequences
    X, y = make_sequences(scaled, n_steps=n_steps)
    print(f"✅ Sequence data created: X={X.shape}, y={y.shape}")

    # Split data 80/20
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Build LSTM model
    model = Sequential([
        LSTM(64, return_sequences=False, input_shape=(n_steps, 1)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    # Train model
    print("🚀 Training started...")
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=2
    )

    # Save model and scaler
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    print("✅ Model and scaler saved successfully!")
    print("Model path:", model_path)
    print("Scaler path:", scaler_path)
    print("🎉 Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/dataset.csv")
    parser.add_argument("--model", default="models/lstm_close_model.h5")
    parser.add_argument("--scaler", default="models/lstm_scaler.pkl")
    parser.add_argument("--n_steps", default=10, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    args = parser.parse_args()

    train(args.csv, args.model, args.scaler, args.n_steps, args.epochs, args.batch_size)
