import os
import pandas as pd

def merge_csv_files(input_dir, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # List all CSV files in the specified directory
    csv_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.csv')]
    
    # Read and concatenate all CSV files
    df_list = []
    for file in csv_files:
        df = pd.read_csv(file)
        df_list.append(df)
    combined_df = pd.concat(df_list, ignore_index=True)

    # Save the combined DataFrame to a CSV file
    combined_df.to_csv(output_file, index=False)
    print(f"Combined CSV created at: {output_file}")

if __name__ == "__main__":
    # Define paths relative to the current file
    current_dir = os.path.dirname(__file__)
    input_dir = os.path.join(current_dir, '..', 'data', 'processed')
    output_file = os.path.join(current_dir, '..', 'data', 'merged', 'current_data.csv')
    
    merge_csv_files(input_dir, output_file)
