import React, { useState } from "react";
import axios from "axios";

const StationCard = ({ station }) => {
  const [prediction, setPrediction] = useState([]);
  const [loading, setLoading] = useState(false);

  const fetchWeatherData = async () => {
    try {
      const response = await axios.get(
        "https://api.open-meteo.com/v1/forecast",
        {
          params: {
            latitude: station.position.lat,
            longitude: station.position.lng,
            hourly: ["temperature_2m", "precipitation_probability"],
            timezone: "auto",
          },
        }
      );

      if (response.status !== 200) {
        throw new Error("Failed to fetch weather data");
      }

      // Assume you need the next 7 hours
      const temperatures = response.data.hourly.temperature_2m.slice(0, 7);
      const precipitationProbabilities =
        response.data.hourly.precipitation_probability.slice(0, 7);

      return { temperatures, precipitationProbabilities };
    } catch (error) {
      console.error("Failed to fetch weather data:", error);
      throw new Error("Failed to fetch weather data");
    }
  };

  const handlePredict = async () => {
    setLoading(true);
    try {
      const { temperatures, precipitationProbabilities } =
        await fetchWeatherData();

      const response = await axios.post("http://localhost:8000/predict", {
        station_name: station.identifier, // Use the correct identifier as needed
        current_time: new Date().toISOString(),
        temperatures_2m: temperatures,
        precipitation_probabilities: precipitationProbabilities,
      });

      // Process predictions to ensure non-negative values
      const adjustedPredictions = response.data.predictions.map((pred) =>
        pred < 0 ? (pred > -4 ? Math.abs(pred) : 0) : pred
      );

      setPrediction(adjustedPredictions);
    } catch (error) {
      console.error("Prediction error:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="station-card">
      <h3>{station.name}</h3>
      <button onClick={handlePredict} disabled={loading}>
        {loading ? "Predicting..." : "Predict"}
      </button>
      {prediction && prediction.length > 0 && (
        <ul>
          {prediction.map((pred, index) => (
            <li key={index}>
              Hour {index + 1}: {pred}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

export default StationCard;
