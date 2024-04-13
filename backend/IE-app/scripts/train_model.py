import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
    model = tensorflow.keras.models.Sequential()
    model.add(tensorflow.keras.layers.LSTM(50, activation='relu', return_sequences=True, input_shape=(n_input, n_features)))
    model.add(tensorflow.keras.layers.LSTM(50, activation='relu'))
    model.add(tensorflow.keras.layers.Dense(1))
    model.compile(optimizer='adam', loss=tensorflow.keras.losses.MeanSquaredError())
    return model

def save_metrics(y_true, y_pred, filename):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    with open(filename, 'w') as file:
        file.write(f'MSE: {mse}\n')
        file.write(f'MAE: {mae}\n')
        file.write(f'R2 Score: {r2}\n')

# Paths
base_dir = os.path.dirname(__file__)
processed_dir = os.path.join(base_dir, '..', 'data', 'processed')
models_dir = os.path.join(base_dir, '..', 'models')
metrics_dir = os.path.join(base_dir, '..', 'reports')

os.makedirs(models_dir, exist_ok=True)
os.makedirs(metrics_dir, exist_ok=True)

n_steps = 3  # Static look-back period
for csv_file in os.listdir(processed_dir):
    if csv_file.endswith('.csv'):
        csv_path = os.path.join(processed_dir, csv_file)
        X_train, X_test, y_train, y_test = preprocess_and_load_data(csv_path, n_steps)
        
        model = build_lstm_model(n_steps, X_train.shape[2])
        history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), verbose=2, callbacks=[EarlyStopping(monitor='val_loss', patience=10)])
        
        # Save the model
        model_name = csv_file.replace('.csv', '.h5')
        model_path = os.path.join(models_dir, model_name)
        model.save(model_path)
        
        # Predict on the training and test set
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Save metrics
        train_metrics_file = os.path.join(metrics_dir, 'train_metrics.txt')
        test_metrics_file = os.path.join(metrics_dir, 'metrics.txt')
        save_metrics(y_train, y_train_pred, train_metrics_file)
        save_metrics(y_test, y_test_pred, test_metrics_file)
        
        print(f'Model {model_name} saved. Metrics reported.')
