import os
import json
import geojson
import pandas as pd
import ast
from decimal import Decimal, getcontext

class GeoJSONFolderConverter:
    def __init__(self, input_folder, output_folder, metadata_csv_path):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.metadata_df = pd.read_csv(metadata_csv_path)
        os.makedirs(self.output_folder, exist_ok=True)

    def read_corners_and_gsd_csv(self, data, json_path):
        try:
            print(json_path)
            image_name = os.path.basename(json_path).replace('.json', '.JPG')
            img_name = image_name[:-10] + ".JPG"
            print(img_name)
            row = data[data['image_name'] == img_name]
            if not row.empty:
                coordinates = ast.literal_eval(row.iloc[0]['corners'])
                gsd = row.iloc[0]['gsd']
                width = int(row.iloc[0]['image_width'])
                height = int(row.iloc[0]['image_height'])
                print(row)
                return coordinates, gsd, width, height, image_name
        except Exception as e:
            print(f"Error reading metadata from {json_path}: {str(e)}")
        return None, None, None, None, None

    def interpolate_to_gps(self, x, y, top_left, bottom_right, width, height):
        lon = top_left[1] + (x / width) * (bottom_right[1] - top_left[1])
        lat = top_left[0] + (y / height) * (bottom_right[0] - top_left[0])
        return lat, lon

    def convert_file(self, json_path):
        getcontext().prec = 48
        corners, gsd, width, height, image_name = self.read_corners_and_gsd_csv(self.metadata_df, json_path)
        if not corners:
            print(f"Skipping {json_path} — missing metadata.")
            return

        top_left = (corners[0][1], corners[0][0])       # NW
        bottom_right = (corners[2][1], corners[2][0])   # SE

        with open(json_path) as f:
            data = json.load(f)

        features = []
        for detection in data.get("detections", []):
            x, y = detection["coordinates"]
            lat, lon = self.interpolate_to_gps(x, y, top_left, bottom_right, width, height)
            point = geojson.Point((lon, lat))
            feature = geojson.Feature(
                geometry=point,
                properties={
                    "name": detection.get("name", "unknown"),
                    "image": image_name,
                    "gsd": gsd
                }
            )
            features.append(feature)

        output_path = os.path.join(self.output_folder, os.path.basename(json_path).replace('.json', '.geojson'))
        feature_collection = geojson.FeatureCollection(features)
        with open(output_path, "w") as f:
            geojson.dump(feature_collection, f, indent=4)

    def convert_all(self):
        for file in os.listdir(self.input_folder):
            if file.endswith(".json"):
                json_path = os.path.join(self.input_folder, file)
                self.convert_file(json_path)


input_folder = r"C:\Users\Administrator\Desktop\Saad_crop_approved_data\canola_field _testing\Canola_seed\2307\final_json" # Directory containing JSON files
output_folder = r"C:\Users\Administrator\Desktop\Saad_crop_approved_data\canola_field _testing\Canola_seed\2307\final_geojson" # Directory containing JSON files
metadata_csv_path = r"C:\Users\Administrator\Desktop\Saad_crop_approved_data\canola_field _testing\Canola_seed\2307\Alligned_images\image_details_chunk1.csv"

converter = GeoJSONFolderConverter(
    input_folder=input_folder,
    output_folder=output_folder,
    metadata_csv_path=metadata_csv_path
)

converter.convert_all()
