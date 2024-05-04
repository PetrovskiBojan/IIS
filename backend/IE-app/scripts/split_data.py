import pandas as pd

def split_dataset(data_path, train_path, test_path, test_size=0.1):
    data = pd.read_csv(data_path)
    
    data['time'] = pd.to_datetime(data['time'])
    data = data.sort_values(by='time', ascending=True)
    
    split_idx = int(len(data) * (1 - test_size))
    
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)

if __name__ == "__main__":
    split_dataset(
        data_path='data/merged/current_data.csv',
        train_path='data/merged/train.csv',
        test_path='data/merged/test.csv',
        test_size=0.1
    )
