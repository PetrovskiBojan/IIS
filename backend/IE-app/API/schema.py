#Schema that wraps the input request object to make it exemplary for database
def individual_serializer(request) -> dict:
    return {
        "station_name": request.station_name,
        "current_time": request.current_time,
        "temperatures_2m": request.temperatures_2m,
        "precipitation_probabilities": request.precipitation_probabilities,
    }