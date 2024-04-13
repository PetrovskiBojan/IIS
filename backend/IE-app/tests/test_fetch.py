import pytest
from unittest.mock import patch
from scripts.fetch_data import fetch_bike_data, fetch_weather_data, combined_dir
import os

# Example responses to mock
mock_bike_response = {
    "number": 123,
    "name": "Station A",
    "position": {"lat": 46.056947, "lng": 14.505751},
    "available_bike_stands": 10
}

mock_weather_response = {
    "hourly": {
        "time": ["2024-03-25T13:00:00"],
        "temperature_2m": [15],
        "precipitation_probability": [20]
    }
}

@pytest.fixture
def cleanup_files():
    # Cleanup before and after tests
    yield
    for filename in os.listdir(combined_dir):
        file_path = os.path.join(combined_dir, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

@patch('requests.get')
def test_fetch_bike_data(mock_get):
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = [mock_bike_response]
    response = fetch_bike_data("dummy_url")
    assert response == [mock_bike_response]

@patch('requests.get')
def test_fetch_weather_data(mock_get):
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = mock_weather_response
    time, temperature, precipitation_probability = fetch_weather_data(46.056947, 14.505751)
    assert time == "2024-03-25T13:00:00"
    assert temperature == 15
    assert precipitation_probability == 20

@patch('requests.get')
def test_combined_data_file_creation(mock_get, cleanup_files):
    mock_get.side_effect = [
        {'json.return_value': [mock_bike_response], 'status_code': 200},
        {'json.return_value': mock_weather_response, 'status_code': 200}
    ]
    # Trigger script's main functionality to generate combined CSV file
    # You need to adjust this part to actually call your script's main function or replicate its logic here

    expected_file_path = os.path.join(combined_dir, f"{mock_bike_response['number']}_{mock_bike_response['name'].replace(' ', '_').replace(',', '').replace('-', '_')}_combined.csv")
    assert os.path.exists(expected_file_path)
    # Further checks can be added here to verify the content of the created file
