import os
import json
import geojson
from PIL import Image
import piexif
import pandas as pd
import traceback
import ast

class GeoJSONConverter:
    def __init__(self, json_path, output_path):
        self.json_path = json_path
        self.output_path = output_path

        corners, gsd, width, height , image_name = self.read_corners(json_path)
        print(corners)
        self.top_left = (corners[1][1], corners[1][0])  # lat, lon
        self.bottom_right = (corners[3][1], corners[3][0])
        self.image_width, self.image_height = width, height

    def read_corners(self, json_path):
        try:
            data = pd.read_csv(r"C:\Users\User\Downloads\image_details (3).csv")
            image_name = os.path.basename(json_path)
            image_name = image_name.replace('.json', '.JPG')
            row = data[data['image_name'] == image_name]
            if not row.empty:
                coordinates = (row.iloc[0]['corners'])
                coordinates = ast.literal_eval(coordinates)
                gsd = row.iloc[0]['gsd']
                width = int(row.iloc[0]['image_width'])
                height = int(row.iloc[0]['image_height'])
                # print(f"this Image: {image_name}, GSD: {gsd}, Width: {width}, Height: {height}, Coordinates: {coordinates}")
                return coordinates, gsd, width, height , image_name
        except Exception as e:
            print(f"Error reading metadata from {json_path}: {str(e)}")
        return None, None, None, None, None

    def interpolate_to_gps(self, x, y):
        lon = self.top_left[1] + (x / self.image_width) * (self.bottom_right[1] - self.top_left[1])
        lat = self.top_left[0] + (y / self.image_height) * (self.bottom_right[0] - self.top_left[0])
        return lat, lon

    def convert_to_geojson(self):
        with open(self.json_path) as f:
            data = json.load(f)

        features = []
        for detection in data.get("detections", []):
            x, y = detection["coordinates"]
            lat, lon = self.interpolate_to_gps(x, y)
            point = geojson.Point((lon, lat))
            feature = geojson.Feature(
                geometry=point,
                properties={"name": detection["name"]}
            )
            features.append(feature)

        feature_collection = geojson.FeatureCollection(features)
        with open(self.output_path, 'w') as geojson_file:
            geojson.dump(feature_collection, geojson_file, indent=4)

def process_folders(json_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(json_folder):
        if filename.endswith(".json"):
            json_path = os.path.join(json_folder, filename)
            with open(json_path, 'r') as f:
                json_data = json.load(f)

            image_name = json_data.get("ImagePath")
            output_name = os.path.splitext(filename)[0] + ".geojson"
            output_path = os.path.join(output_folder, output_name)

            try:
                converter = GeoJSONConverter(json_path, output_path)
                converter.convert_to_geojson()
                print(f"Converted: {filename} -> {output_name}")
            except Exception as e:
                print(f"Failed to process {filename}: {e}")
                print(traceback.format_exc())

# Example usage:
image_folder = r"C:\Users\User\Downloads\rotated_jsons\rotated_jsons"
json_folder = r"C:\Users\User\Downloads\DJI_20250424121519_0004_D_LA"
output_folder = r"C:\Users\User\Downloads\rotated_jsons\rotated_result"

os.makedirs(output_folder, exist_ok=True)


process_folders(json_folder, output_folder)
