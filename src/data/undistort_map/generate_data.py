import numpy as np

# # This generates a smiley face :))
# longitudes = [
#     -10,
#     10, 
#     -5, 
#     -2, 
#     0, 
#     2, 
#     5, 
# ]
# latitudes = [
#     30,
#     30,
#     0,
#     -5,
#     -8,
#     -5,
#     0,
# ]
 
# Generate uniform random samples for latitude and longitude
num_samples = 100
latitudes = np.random.uniform(-90, 90, num_samples)
longitudes = np.random.uniform(-180, 180, num_samples)

# Combine latitudes and longitudes into a single array
coordinates = np.column_stack((latitudes, longitudes))

# Save the array to an .npy file
np.save('coordinates.npy', coordinates)