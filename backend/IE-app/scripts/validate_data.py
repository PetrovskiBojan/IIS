import pandas as pd
from great_expectations.dataset import PandasDataset

def validate_data(csv_path):
    df = pd.read_csv(csv_path, parse_dates=['time'])
    df['time'] = df['time'].dt.floor('Min')
    
    ge_df = PandasDataset(df)
    # Set expectations
    ge_df.expect_column_values_to_not_be_null('temperature_2m')
    ge_df.expect_column_values_to_be_between('temperature_2m', minimum=-50, maximum=50)
    ge_df.expect_column_values_to_not_be_null('precipitation_probability')
    ge_df.expect_column_values_to_be_between('precipitation_probability', minimum=0, maximum=100)
    
    # Validate data
    results = ge_df.validate(result_format='SUMMARY')
    if not results["success"]:
        print("Data validation failed. Please check the dataset.")
        return False
    print("Data validation passed.")
    return True

if __name__ == "__main__":
    import sys
    csv_path = sys.argv[0]
    if not validate_data(csv_path):
        sys.exit(1) 