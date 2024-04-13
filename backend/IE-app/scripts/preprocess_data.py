import json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

def preprocess_data(csv_path, processed_dir, scaler_params_dir):
    # Load the data
    df = pd.read_csv(csv_path, parse_dates=['time'])
    
    # Simplify the 'time' to remove seconds and milliseconds
    df['time'] = df['time'].dt.floor('Min')
    
    # Normalize 'temperature_2m' and 'precipitation_probability' to range [0, 1]
    scaler = MinMaxScaler()
    df[['temperature_2m', 'precipitation_probability']] = scaler.fit_transform(df[['temperature_2m', 'precipitation_probability']])
    
    # Save the preprocessed data to the 'processed' directory
    processed_path = os.path.join(processed_dir, os.path.basename(csv_path))
    df.to_csv(processed_path, index=False)
    print(f"Preprocessed data saved to {processed_path}")
    
    # Save the scaler parameters to JSON
    scaler_params_path = os.path.join(scaler_params_dir, os.path.basename(csv_path).replace('.csv', '_scaler_params.json'))
    with open(scaler_params_path, 'w') as f:
        json.dump({'min_': scaler.min_.tolist(), 'scale_': scaler.scale_.tolist()}, f)
    print(f"Scaler parameters saved to {scaler_params_path}")

if __name__ == "__main__":
    # Specify the directories
    current_dir = os.path.dirname(__file__)
    combined_dir = os.path.abspath(os.path.join(current_dir, '..', 'data', 'combined'))
    processed_dir = os.path.abspath(os.path.join(current_dir, '..', 'data', 'processed'))
    scaler_params_dir = os.path.abspath(os.path.join(current_dir, '..', 'data', 'scaler_params'))
    
    # Ensure the 'processed' and 'scaler_params' directories exist
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(scaler_params_dir, exist_ok=True)
    
    # Process each CSV file in the 'combined' directory
    for filename in os.listdir(combined_dir):
        if filename.endswith(".csv"):
            print(f"Processing {filename}...")
            csv_path = os.path.join(combined_dir, filename)
            preprocess_data(csv_path, processed_dir, scaler_params_dir)
