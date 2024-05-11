from pydantic import BaseModel

class PredictionRequest(BaseModel):
    station_name: str
    current_time: str 
    temperatures_2m: list[float]
    precipitation_probabilities: list[float]