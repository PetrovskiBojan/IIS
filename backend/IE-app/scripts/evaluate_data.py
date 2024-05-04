import pandas as pd
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def generate_data_drift_report(current_data_path, reference_data_path, report_path):
    # Load the data
    current_data = pd.read_csv(current_data_path)
    reference_data = pd.read_csv(reference_data_path)

    # Column mapping, if your data needs it (optional)
    column_mapping = ColumnMapping()

    # Create a data drift report
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)
    report.save_html(report_path)

if __name__ == "__main__":
    generate_data_drift_report("data/merged/current_data.csv", "data/merged/reference_data.csv", "reports/data_drift_report.html")
