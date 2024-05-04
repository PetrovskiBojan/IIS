import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import mlflow
import mlflow.tensorflow
from dotenv import load_dotenv
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Load environment variables from .env file
load_dotenv()

# Setup MLflow to track to DagsHub
mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
mlflow.set_experiment('Bike_Sharing_Demand_Forecasting')

def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        seq_x, seq_y = data[i:(i + n_steps), :-1], data[i + n_steps, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def preprocess_and_load_data(csv_file, n_steps=3):
    df = pd.read_csv(csv_file, index_col='time', parse_dates=True)
    df.sort_index(inplace=True)
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df.values)
    
    X, y = create_sequences(scaled_data, n_steps)
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def build_lstm_model(n_input, n_features):
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(n_input, n_features)),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def main():
    # Define paths
    base_dir = os.path.dirname(__file__)
    processed_dir = os.path.join(base_dir, '..', 'data', 'processed')

    n_steps = 3  # Static look-back period
    for csv_file in os.listdir(processed_dir):
        if csv_file.endswith('.csv'):
            csv_path = os.path.join(processed_dir, csv_file)
            X_train, X_test, y_train, y_test = preprocess_and_load_data(csv_path, n_steps)

            with mlflow.start_run():
                mlflow.set_tag("csv_file", csv_file)
                model = build_lstm_model(n_steps, X_train.shape[2])
                history = model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test), verbose=2, callbacks=[EarlyStopping(monitor='val_loss', patience=10)])
                
                # Logging metrics
                train_mse = history.history['loss'][-1]
                test_mse = history.history['val_loss'][-1]
                mlflow.log_metric("train_mse", train_mse)
                mlflow.log_metric("test_mse", test_mse)

                # Predict and calculate additional metrics
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                train_mae = mean_absolute_error(y_train, y_train_pred)
                test_mae = mean_absolute_error(y_test, y_test_pred)
                train_r2 = r2_score(y_train, y_train_pred)
                test_r2 = r2_score(y_test, y_test_pred)
                mlflow.log_metric("train_mae", train_mae)
                mlflow.log_metric("test_mae", test_mae)
                mlflow.log_metric("train_r2", train_r2)
                mlflow.log_metric("test_r2", test_r2)

                # Save and log the model within the MLflow
                mlflow.keras.log_model(model, "model")
                print(f'Model {csv_file.replace(".csv", ".h5")} saved. Metrics reported.')

if __name__ == "__main__":
    main()
