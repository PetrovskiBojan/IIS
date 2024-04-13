import React from "react";
import StationCard from "./components/StationCard";
import { STATION_NAMES } from "./constants/stations";
import "./styles/style.css";

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>Station Predictions</h1>
      </header>
      <main className="main">
        {STATION_NAMES.map((station) => (
          <StationCard key={station.id} station={station} />
        ))}
      </main>
    </div>
  );
}

export default App;
