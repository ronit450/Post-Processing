import numpy as np
import cv2
import shapefile
import json 

with open(r"C:\Users\User\Downloads\Aiman-file\DJI_20240228083728_0005_D.json") as f:
    data = json.load(f)

# Sample data with corner detections

# Image path and output shapefile path
image_path = r"C:\Users\User\Downloads\copy_DJI_20240228083728_0005_D_aligned_xmp.tif"
output_path = r"C:\Users\User\Downloads\Aiman-file-results\temp\corners_output"

# Define corners (longitude, latitude)
corners = [[-83.08489893270443, 30.123174014690044], 
           [-83.08453339472855, 30.12318037458068], 
           [-83.08452791372119, 30.122942291025208], 
           [-83.08489345082155, 30.122935931191176]]

# Known rotation angle in degrees (make sure this value is correct for your image)
angle = 0.017396958532605443

# Function to rotate a point around the center of the image
def rotate_point(x, y, angle, image_width, image_height):
    angle_rad = np.deg2rad(angle)
    center_x, center_y = image_width / 2, image_height / 2

    # Translate point to origin
    translated_x = x - center_x
    translated_y = y - center_y

    # Apply rotation matrix
    rotated_x = translated_x * np.cos(angle_rad) - translated_y * np.sin(angle_rad)
    rotated_y = translated_x * np.sin(angle_rad) + translated_y * np.cos(angle_rad)

    # Translate point back
    rotated_x += center_x
    rotated_y += center_y

    return rotated_x, rotated_y

# Read the image to get dimensions
image = cv2.imread(image_path)
image_height, image_width, _ = image.shape

# Rotate all four corners to get the new boundaries
rotated_corners = [rotate_point(x * image_width, y * image_height, angle, image_width, image_height) 
                   for x, y in [(0, 0), (1, 0), (1, 1), (0, 1)]]

# Assign rotated corners to corresponding GPS coordinates
gps_corners = {
    "top_left": corners[0],
    "top_right": corners[1],
    "bottom_right": corners[2],
    "bottom_left": corners[3]
}

# Function to interpolate pixel coordinates to GPS coordinates using all four corners
def interpolate_to_gps(x, y):
    # Normalize pixel coordinates between 0 and 1
    norm_x = x / image_width
    norm_y = y / image_height
    
    # Bilinear interpolation for longitude and latitude
    lon = (
        gps_corners["top_left"][0] * (1 - norm_x) * (1 - norm_y) +
        gps_corners["top_right"][0] * norm_x * (1 - norm_y) +
        gps_corners["bottom_right"][0] * norm_x * norm_y +
        gps_corners["bottom_left"][0] * (1 - norm_x) * norm_y
    )
    
    lat = (
        gps_corners["top_left"][1] * (1 - norm_x) * (1 - norm_y) +
        gps_corners["top_right"][1] * norm_x * (1 - norm_y) +
        gps_corners["bottom_right"][1] * norm_x * norm_y +
        gps_corners["bottom_left"][1] * (1 - norm_x) * norm_y
    )
    
    return lat, lon

# Create a shapefile writer for POINT type
shp_writer = shapefile.Writer(output_path, shapefile.POINT)
shp_writer.field('Name', 'C')
shp_writer.field('Confidence', 'F', decimal=2)

# Iterate over detections and add them as points to the shapefile
for detection in data['detections']:
    x1, y1 = detection['box']['x1'], detection['box']['y1']
    x2, y2 = detection['box']['x2'], detection['box']['y2']
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    lat, lon = interpolate_to_gps(center_x, center_y)
    shp_writer.point(lon, lat)
    shp_writer.record(detection['name'], detection['confidence'])

# Close the shapefile writer
shp_writer.close()
