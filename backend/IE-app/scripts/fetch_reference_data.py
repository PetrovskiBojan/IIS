import subprocess
import os

def update_reference_data():
    # Paths for the data files
    current_data_path = 'data/merged/current_data.csv'
    reference_data_path = 'data/merged/reference_data.csv'

    # Fetch the last committed version of current_data.csv from DVC
    subprocess.run(['dvc', 'pull', current_data_path], check=True)
    
    # Rename this version as the reference data
    os.rename(current_data_path, reference_data_path)
    print("Reference data updated.")

    # Restore the latest current data
    subprocess.run(['dvc', 'checkout', current_data_path], check=True)
    print("Latest current data restored.")

if __name__ == "__main__":
    update_reference_data()
