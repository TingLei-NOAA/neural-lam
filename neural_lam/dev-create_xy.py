import yaml
import numpy as np
from pyproj import Transformer

# Example latitude and longitude arrays (replace with your actual data)
lon = np.array([...])  # 2D array of longitudes
lat = np.array([...])  # 2D array of latitudes

# Load the YAML projection setup (assuming it's in 'config.yaml')
with open('config.yaml', 'r') as file:
    yaml_data = yaml.safe_load(file)

# Access the projection parameters directly from the dictionary
central_longitude = yaml_data['projection']['kwargs']['central_longitude']
central_latitude = yaml_data['projection']['kwargs']['central_latitude']
standard_parallels = yaml_data['projection']['kwargs']['standard_parallels']

# Create the CRS (Coordinate Reference System) string based on the YAML configuration
mycrs = (f"+proj=lcc +lat_0={central_latitude} +lon_0={central_longitude} "
         f"+lat_1={standard_parallels[0]} +lat_2={standard_parallels[1]} "
         f"+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")

# Set up the Transformer from WGS84 (lat/lon) to Lambert Conformal Conic projection
crstrans = Transformer.from_crs("EPSG:4326", mycrs, always_xy=True)

# Perform the coordinate transformation
x, y = crstrans.transform(lon, lat)

# Combine the x and y coordinates into a single array
xy = np.array([x, y])  # Shape will be (2, n_lon, n_lat)

# Save the transformed coordinates to a .npy file
np.save("grid_xy_coordinates.npy", xy)

print("Transformation complete. Coordinates saved to 'grid_xy_coordinates.npy'.")

