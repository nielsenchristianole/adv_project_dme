
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
    PLOT_ARGS = {"hamlet" : {"color" : "black", "s" : 1},
                 "town" : {"color" : "blue", "s" : 1},
                 "village" : {"color" : "green", "s" : 1},
                 "city" : {"color" : "red", "s" : 1},
                 "other" : {"color" : "yellow", "s" : 1, "marker" : "x"}}
    
    for city_type in PLOT_ARGS.keys():
        
        data = city_data[city_data["type"] == city_type]
        
        ax.scatter(data["lat"], data["lon"], **PLOT_ARGS[city_type])
        
        # for i, row in data.iterrows():
        #     ax.annotate(row["name"], (row["lat"], row["lon"]), ha='center')
        
            

def plot_border(ax : Axis, contour_points):
    
    ax.plot(contour_points)
    
if __name__ == "__main__":
    
    cities = pd.read_csv("./data/cities.csv")
    height_map = plt.imread("./test_heightmap.png")
    height_map = np.mean(height_map, axis=2).T

    cities = utils.get_cities_by_country(cities, "India")

    
    fig, ax = plt.subplots()
    
    plot_height_surface(ax, height_map)
    plot_cities(ax, cities)
    
    ax.axis("equal")
    
    plt.show()