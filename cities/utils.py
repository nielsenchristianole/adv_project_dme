
import pandas as pd

CITY_TYPES = ["hamlet", "town", "village", "city", "other"]

def read_data(file_name):
    return pd.read_csv(file_name)

def get_cities_in_range(data, lat_min, lat_max, lon_min, lon_max, city_types = CITY_TYPES):
        return data[(data["lat"] >= lat_min) & (data["lat"] <= lat_max) & 
                    (data["lon"] >= lon_min) & (data["lon"] <= lon_max) & 
                    (data["type"].isin(city_types))]
        
def get_cities_by_country(data, country):
    return data[data["country"] == country]