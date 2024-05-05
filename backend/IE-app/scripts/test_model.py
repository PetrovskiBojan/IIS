import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import mlflow
import mlflow.tensorflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

# Load environment variables from .env file
load_dotenv()

# Setup MLflow to track to DagsHub
mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
mlflow.set_experiment('Bike_Sharing_Demand_Forecasting')

client = MlflowClient()

def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        seq_x, seq_y = data[i:(i + n_steps), :-1], data[i + n_steps, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def preprocess_and_load_data(train_file, test_file, n_steps=3):
    df_train = pd.read_csv(train_file, index_col='time', parse_dates=True)
    df_test = pd.read_csv(test_file, index_col='time', parse_dates=True)
    df_train.sort_index(inplace=True)
    df_test.sort_index(inplace=True)
    
    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(df_train.values)
    scaled_test = scaler.transform(df_test.values)
    
    X_train, y_train = create_sequences(scaled_train, n_steps)
    X_test, y_test = create_sequences(scaled_test, n_steps)
    
    return X_train, X_test, y_train, y_test

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
    train_file = os.path.join(base_dir, '..', 'data', 'merged', 'train.csv')
    test_file = os.path.join(base_dir, '..', 'data', 'merged', 'test.csv')

    n_steps = 3  # Static look-back period
    X_train, X_test, y_train, y_test = preprocess_and_load_data(train_file, test_file, n_steps)

    with mlflow.start_run() as run:
        model = build_lstm_model(n_steps, X_train.shape[2])
        history = model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test), verbose=2, callbacks=[EarlyStopping(monitor='val_loss', patience=10)])
        
        # Log metrics
        mlflow.log_metric("train_mse", history.history['loss'][-1])
        mlflow.log_metric("test_mse", history.history['val_loss'][-1])

        # Predict and calculate additional metrics
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        mlflow.log_metric("train_mae", mean_absolute_error(y_train, y_train_pred))
        mlflow.log_metric("test_mae", mean_absolute_error(y_test, y_test_pred))
        mlflow.log_metric("train_r2", r2_score(y_train, y_train_pred))
        mlflow.log_metric("test_r2", r2_score(y_test, y_test_pred))

        # Register and evaluate the model
        model_uri = mlflow.keras.log_model(model, "model", registered_model_name="Bike_Sharing_Demand_Forecasting")
        latest_prod_model = client.get_latest_versions("Bike_Sharing_Demand_Forecasting", stages=["Production"])
        
        if latest_prod_model:
            best_mse = latest_prod_model[0].metrics['test_mse']
            if history.history['val_loss'][-1] < best_mse:
                client.transition_model_version_stage(
                    name="Bike_Sharing_Demand_Forecasting",
                    version=run.info.run_id,
                    stage="Production",
                    archive_existing_versions=True
                )

        print("Model training and testing complete. Metrics reported.")

if __name__ == "__main__":
    main()
