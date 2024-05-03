import os
import pandas as pd
import json
import subprocess
from sklearn.preprocessing import MinMaxScaler

def run_dvc_pull(data_directory):
    """Pull the latest data from DVC for the specified directory."""
    print("Pulling latest data from DVC...")
    try:
        subprocess.run(['dvc', 'pull', data_directory], check=True)
        print("Data pulled successfully.")
    except subprocess.CalledProcessError as e:
        print("Failed to pull data from DVC:", e)
        raise

def preprocess_data(file_path, processed_dir, scaler_params_dir):
    """Load, preprocess, and save the data."""
    print(f"Preprocessing data for file: {file_path}")
    df = pd.read_csv(file_path)

    # Normalize specific columns
    if 'temperature_2m' in df.columns and 'precipitation_probability' in df.columns:
        scaler = MinMaxScaler()
        df[['temperature_2m', 'precipitation_probability']] = scaler.fit_transform(df[['temperature_2m', 'precipitation_probability']])
    
        # Save the scaler parameters for potential inverse transformation or future use
        scaler_params = {
            'min_': scaler.min_.tolist(),
            'scale_': scaler.scale_.tolist(),
            'data_min_': scaler.data_min_.tolist(),
            'data_max_': scaler.data_max_.tolist(),
            'data_range_': scaler.data_range_.tolist()
        }
        scaler_filename = os.path.basename(file_path).replace('.csv', '_scaler_params.json')
        scaler_filepath = os.path.join(scaler_params_dir, scaler_filename)
        with open(scaler_filepath, 'w') as f:
            json.dump(scaler_params, f)
        print(f"Scaler parameters saved to: {scaler_filepath}")

    # Save the preprocessed data
    processed_filepath = os.path.join(processed_dir, os.path.basename(file_path))
    df.to_csv(processed_filepath, index=False)
    print(f"Processed data saved to: {processed_filepath}")

if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    data_dir = os.path.join(current_dir, '..', 'data', 'combined')
    processed_dir = os.path.join(current_dir, '..', 'data', 'processed')
    scaler_params_dir = os.path.join(current_dir, '..', 'data', 'scaler_params')

    # Ensure the output directories exist
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(scaler_params_dir, exist_ok=True)

    # Pull the latest data from DVC
    run_dvc_pull(data_dir)

    # Process each CSV file in the data directory
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):  # Make sure to process only CSV files
            file_path = os.path.join(data_dir, filename)
            preprocess_data(file_path, processed_dir, scaler_params_dir)
