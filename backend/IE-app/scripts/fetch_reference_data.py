import subprocess
import shutil
import os

def update_reference_data():
    # Paths for the data files
    current_data_path = 'data/merged/current_data.csv'
    reference_data_path = 'data/merged/reference_data.csv'

    # Fetch the last committed version of current_data.csv from DVC
    subprocess.run(['dvc', 'pull', current_data_path], check=True)
    
    if os.path.exists(reference_data_path):
        os.remove(reference_data_path)
    shutil.move(current_data_path, reference_data_path)
    print("Reference data updated.")

    # Add the updated reference data to DVC
    subprocess.run(['dvc', 'add', reference_data_path], check=True)

    # Restore the latest current data
    subprocess.run(['dvc', 'checkout', current_data_path], check=True)
    print("Latest current data restored.")

    # Push the updated reference data to the remote storage
    subprocess.run(['dvc', 'push', '-r', 'origin'], check=True)
    print("Reference data pushed to DVC remote.")

    # Add all changes to git
    subprocess.run(['git', 'add', '.'], check=True)
    # Commit the changes
    subprocess.run(['git', 'commit', '-m', 'Update of reference and current data'], check=True)
    # Push the changes to the git remote repository
    subprocess.run(['git', 'push'], check=True)
    print("Changes pushed to Git remote.")

if __name__ == "__main__":
    update_reference_data()
