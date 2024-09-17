import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

"""
File to test orthographic projection and mercator projection of lat lon coords
"""

# Coordinates for the smiley face (latitudes and longitudes)
latitudes = [30, 30, 0, -5, -8, -5, 0]
longitudes = [-10, 10, 5, 2, 0, -2, -5]

def plot_mercator(latitudes, longitudes):
    plt.figure(figsize=(8, 8))
    
    # Create the Mercator projection map
    m = Basemap(projection='merc', llcrnrlat=-60, urcrnrlat=60, llcrnrlon=-20, urcrnrlon=20, resolution='i')
    
    # Draw coastlines, countries, and continents
    m.drawcoastlines()
    m.drawcountries()
    m.fillcontinents(color='lightgray', lake_color='aqua')
    
    # Draw parallels and meridians
    m.drawparallels(range(-60, 61, 10), labels=[1, 0, 0, 0])
    m.drawmeridians(range(-20, 21, 10), labels=[0, 0, 0, 1])

    # Plot the points (the smiley face) on the Mercator map
    x, y = m(longitudes, latitudes)
    m.plot(x, y, 'bo', markersize=10)  # Blue dots for points
    m.plot(x[2:7], y[2:7], 'r-', linewidth=2)  # Red line for mouth curve
    
    # Title
    plt.title("Smiley Face on Earth (Mercator Projection)")
    plt.show()

def plot_orthographic(latitudes, longitudes):
    plt.figure(figsize=(8, 8))
    
    # Create the Orthographic projection map (centered near 0 longitude and 0 latitude)
    m = Basemap(projection='ortho', lat_0=0, lon_0=0, resolution='l')
    
    # Draw coastlines, countries, and continents
    m.drawcoastlines()
    m.drawcountries()
    m.fillcontinents(color='lightgray', lake_color='aqua')
    
    # Plot the points (the smiley face) on the Orthographic map
    x, y = m(longitudes, latitudes)
    m.plot(x, y, 'bo', markersize=10)  # Blue dots for points
    m.plot(x[2:7], y[2:7], 'r-', linewidth=2)  # Red line for mouth curve
    
    # Title
    plt.title("Smiley Face on Earth (Orthographic Projection)")
    plt.show()

# Plot in Mercator projection
plot_mercator(latitudes, longitudes)

# Plot in Orthographic projection
plot_orthographic(latitudes, longitudes)
