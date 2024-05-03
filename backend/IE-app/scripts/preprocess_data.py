import os
import pandas as pd
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

def preprocess_data(file_path, processed_dir):
    """Load, preprocess, and save the data."""
    print(f"Preprocessing data for file: {file_path}")
    df = pd.read_csv(file_path)

    # Example of preprocessing: Normalize specific columns
    # Ensure to replace 'column_to_normalize' with your actual column names
    if 'temperature_2m' in df.columns and 'precipitation_probability' in df.columns:
        scaler = MinMaxScaler()
        df[['temperature_2m', 'precipitation_probability']] = scaler.fit_transform(df[['temperature_2m', 'precipitation_probability']])
    
    # Save the preprocessed data
    processed_filepath = os.path.join(processed_dir, os.path.basename(file_path))
    df.to_csv(processed_filepath, index=False)
    print(f"Processed data saved to: {processed_filepath}")

if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    data_dir = os.path.join(current_dir, '..', 'data', 'combined')
    processed_dir = os.path.join(current_dir, '..', 'data', 'processed')

    # Ensure the output directory exists
    os.makedirs(processed_dir, exist_ok=True)

    # Pull the latest data from DVC
    run_dvc_pull(data_dir)

    # Process each CSV file in the data directory
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):  # Make sure to process only CSV files
            file_path = os.path.join(data_dir, filename)
            preprocess_data(file_path, processed_dir)
