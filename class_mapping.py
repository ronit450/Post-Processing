import os
import pandas as pd
import geopandas as gpd
import json
import shutil

# Paths
tiff_folder = r"C:\Users\User\Downloads\Aiman-ko-use-karna-he\Aiman-ko-use-karna-he\clipped"  # Folder containing TIFF files
geojson_folder = r"C:\Users\User\Downloads\Aiman-ko-use-karna-he\Aiman-ko-use-karna-he\geojson"  # Folder containing GeoJSON files
csv_path = r"C:\Users\User\Downloads\Aiman-ko-use-karna-he\Aiman-ko-use-karna-he\Training_class_id.csv"  # Path to CSV file
output_folder = r"C:\Users\User\Downloads\Aiman-ko-use-karna-he"  # Output folder where folders will be created for each class

# Step 1: Load the CSV for class mapping
class_df = pd.read_csv(csv_path)

# Step 2: Function to get class name from the CSV
def get_class_name(site_name, class_num):
    class_row = class_df[(class_df['Study site name'] == site_name) & (class_df['class'] == class_num)]
    if not class_row.empty:
        return class_row.iloc[0]['Label']  # Fetch the class name (label) from the CSV
    else:
        return None

# Step 3: Process TIFF files
for tiff_file in os.listdir(tiff_folder):
    if tiff_file.endswith(".tif"):
        # Extract site name and feature (class number) from the file name
        parts = tiff_file.split('_')
        site_name = parts[1]  # Extract site name
        class_number = int(parts[2])
        feature_index = int(parts[-1].split('f')[1].split('.')[0])  # Extract feature number (class number)
        
        # Load corresponding GeoJSON file
        geojson_file = f"bb_points_Ds_training_{site_name}.geojson"
        geojson_path = os.path.join(geojson_folder, geojson_file)
        
        if os.path.exists(geojson_path):
            geojson_data = gpd.read_file(geojson_path)
            
            # Get the corresponding class name from the CSV
            class_name = get_class_name(site_name, class_number)  # CSV class number starts from 1, feature index from 0
            
            if class_name:
                # Create output folder for the class name
                class_folder = os.path.join(output_folder, class_name)
                os.makedirs(class_folder, exist_ok=True)
                
                # Copy the TIFF file to the class folder
                shutil.copy(os.path.join(tiff_folder, tiff_file), class_folder)
                
                # Step 4: Create corresponding JSON file
                output_json = {
                    "type": "FeatureCollection",
                    "features": []
                }
                
                # Extract the relevant feature (bounding boxes) from the GeoJSON
                if feature_index < len(geojson_data):
                    bbox_feature = geojson_data.iloc[feature_index]  # Use the correct feature based on the feature index
                    feature_data = {
                        "type": "Feature",
                        "properties": {
                            "class": class_name
                        },
                        "geometry": bbox_feature['geometry'].__geo_interface__
                    }
                    output_json['features'].append(feature_data)
                
                # Save the JSON file with the same name as the TIFF file
                json_filename = os.path.join(class_folder, tiff_file.replace('.tif', '.json'))
                with open(json_filename, 'w') as f:
                    json.dump(output_json, f, indent=4)
                
            else:
                print(f"Class name not found for site: {site_name} and class number: {feature_index + 1}")
        else:
            print(f"GeoJSON file for {site_name} not found.")
