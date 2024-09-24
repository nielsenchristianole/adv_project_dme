
from matplotlib import pyplot as plt
from matplotlib.axis import Axis
import numpy as np

import cities.utils as utils

import pandas as pd

def plot_height_surface(ax : Axis, height_data):
    """
    Plot a (x,y) grid of 32-bit floats as a countour plot.
    """
    
    ax.contour(height_data, levels=5)

def plot_cities(ax : Axis, city_data : pd.DataFrame):    
    PLOT_ARGS = {"hamlet" : {"color" : "black", "s" : 1, "marker" : "x"},
                 "town" : {"color" : "blue", "s" : 20},
                 "village" : {"color" : "green", "s" : 30},
                 "city" : {"color" : "red", "s" : 40},
                 "other" : {"color" : "yellow", "s" : 10, "marker" : "x"}}
    
    for city_type in PLOT_ARGS.keys():
        
        data = city_data[city_data["type"] == city_type]
        
        ax.scatter(data["lat"], data["lon"], **PLOT_ARGS[city_type])
        
        # for i, row in data.iterrows():
        #     ax.annotate(row["name"], (row["lat"], row["lon"]), ha='center')
        
            

def plot_border(ax : Axis, contour_points):
    
    ax.plot(contour_points)
    
    
if __name__ == "__main__":
    from pathlib import Path
    import geopandas as gpd
    import tqdm
    
    esri_dir = Path('data/vectors/')
    path = esri_dir / 'ne_10m_coastline/ne_10m_coastline.shp'
    gdf = gpd.read_file(path)
    cities = pd.read_csv("./data/cities.csv")
    fig, ax = plt.subplots()
    plot_cities(ax, cities)
    # plt.show()
    is_closed = gdf["geometry"].apply(lambda x : x.is_closed)
    for i in tqdm.tqdm(np.argwhere(is_closed).squeeze()):
        x,y = gdf["geometry"][i].xy
        
        # cities_ = utils.get_cities_in_range(cities, np.min(x), np.max(x), np.min(y), np.max(y))
        # cities = utils.get_cities_by_country(cities, "Madagasikara / Madagascar", ["town"])
        # print(cities["country"].unique())
        
        
        ax.plot(x,y)
        
    ax.axis("equal")
        
        # plt.savefig(f"./out/cities_{i}.png")
    plt.show()
    plt.close()