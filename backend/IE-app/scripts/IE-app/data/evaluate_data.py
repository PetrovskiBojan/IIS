import pandas as pd
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab
import json

def evaluate_data(reference_data_path, current_data_path, report_path):
    reference_data = pd.read_csv(reference_data_path)
    current_data = pd.read_csv(current_data_path)
    
    data_drift_dashboard = Dashboard(tabs=[DataDriftTab()])
    data_drift_dashboard.calculate(reference_data, current_data, column_mapping=None)
    
    report = data_drift_dashboard.show()
    report.save(report_path)
    print(f"Data drift report saved to {report_path}")

if __name__ == "__main__":
    import sys
    reference_data_path = sys.argv[1]
    current_data_path = sys.argv[2]
    report_path = sys.argv[3]
    evaluate_data(reference_data_path, current_data_path, report_path)
