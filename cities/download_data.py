import matplotlib.pyplot as plt
import numpy as np
import json
import tqdm
from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/cities/")
CITY_TYPES = ["hamlet", "town", "village", "city", "other"]

import requests, zipfile, io, re

# find all links that ends with .zip
def find_zip_links(url):
    page = requests.get(url).text
    links = re.findall(r'href=[\'"]?([^\'" >]+)', page)
    return [url + link for link in links if link.endswith('.zip')]

def extract_zip_files(url):
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall("data/cities")
    
def get_and_download_data():
    url = "https://www.geoapify.com/data-share/localities/"
    zip_links = find_zip_links(url)
    for link in zip_links:
        extract_zip_files(link)
    
# Extract all the data from the .ndjson files.
def load_ndjson(file):
    data = []
    with open(file, "r", encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def convert_to_pandas():
    
    data_paths = list(Path(DATA_PATH).rglob("*.ndjson"))
    
    data = []
    for i, file_name in enumerate(tqdm.tqdm(data_paths, total=len(data_paths), desc="Converting to pandas")):
        cities =  load_ndjson(file_name)
        for city in cities:
            city_type = city.get("type", "other")
            location = city["location"]
            name = city.get("name", "Unknown")
            if "address" in city:
                if city["address"] == []:
                    country = "Unknown"
                else:
                    country = city["address"].get("country", "Unknown")
            else:
                country = "Unknown"
            bbox = city["bbox"]
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            data.append({"country" : country,
                         "name" : name, 
                         "type": city_type, 
                         "lat": location[0], 
                         "lon": location[1],
                         "bbox_area": area})
        
    print("(1/2) Writing to pandas dataframe")
    df = pd.DataFrame(data)
    
    print("(2/2) Saving to .csv")
    df.to_csv("data/cities.csv", index=False)

if __name__ == "__main__":
    get_and_download_data()
    convert_to_pandas()
