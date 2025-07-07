import os
import json
import pandas as pd
import ast
from decimal import Decimal, getcontext
from geopy.distance import geodesic

class GeoJSONFolderConverter:
    def __init__(self, input_folder, output_folder, metadata_csv_path):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.metadata_df = pd.read_csv(metadata_csv_path)
        os.makedirs(self.output_folder, exist_ok=True)
        getcontext().prec = 48

    def read_corners_and_gsd_csv(self, data, json_path):
        try:
            json_filename = os.path.basename(json_path).replace('.json', '.JPG')
            matching_row = data[data['image_name'] == json_filename]
            if not matching_row.empty:
                row = matching_row.iloc[0]
                coordinates = ast.literal_eval(row['corners'])
                gsd = row['gsd']
                width = int(row['image_width'])
                height = int(row['image_height'])
                print(coordinates)
                return coordinates, gsd, width, height, json_filename
        except Exception as e:
            print(f"Error reading metadata from {json_path}: {str(e)}")
        return None, None, None, None, None

    def interpolate_to_gps(self, x, y, gps_corners, width, height):
        norm_x = Decimal(x) / Decimal(width)
        norm_y = Decimal(y) / Decimal(height)

        lon = (
            Decimal(gps_corners["top_left"][1]) * (1 - norm_x) * (1 - norm_y) +
            Decimal(gps_corners["top_right"][1]) * norm_x * (1 - norm_y) +
            Decimal(gps_corners["bottom_right"][1]) * norm_x * norm_y +
            Decimal(gps_corners["bottom_left"][1]) * (1 - norm_x) * norm_y
        )

        lat = (
            Decimal(gps_corners["top_left"][0]) * (1 - norm_x) * (1 - norm_y) +
            Decimal(gps_corners["top_right"][0]) * norm_x * (1 - norm_y) +
            Decimal(gps_corners["bottom_right"][0]) * norm_x * norm_y +
            Decimal(gps_corners["bottom_left"][0]) * (1 - norm_x) * norm_y
        )

        return f"{lat:.48f}", f"{lon:.48f}"

    def convert_file(self, json_path):
        with open(json_path) as f:
            data = json.load(f)

        corners, gsd, width, height, image_name = self.read_corners_and_gsd_csv(self.metadata_df, json_path)
        if not corners:
            print(f"Skipping {json_path} — missing metadata.")
            return

        gps_corners = {
            "top_left": (corners[0][1], corners[0][0]),
            "top_right": (corners[1][1], corners[1][0]),
            "bottom_right": (corners[2][1], corners[2][0]),
            "bottom_left": (corners[3][1], corners[3][0])
        }

        features = []
        for shape in data.get("shapes", []):
            if shape.get("shape_type") != "rectangle" or len(shape.get("points", [])) != 2:
                continue

            (x1, y1), (x2, y2) = shape["points"]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            lat_str, lon_str = self.interpolate_to_gps(center_x, center_y, gps_corners, width, height)

            feature = [
                '    {',
                '      "type": "Feature",',
                '      "geometry": {',
                '        "type": "Point",',
                f'        "coordinates": [{lon_str}, {lat_str}]',
                '      },',
                '      "properties": {',
                f'        "label": "{shape.get("label", "unknown")}",',
                f'        "image": "{image_name}",',
                f'        "gsd": {gsd}',
                '      }',
                '    }'
            ]
            features.append('\n'.join(feature))

        features_str = ',\n'.join(features)
        geojson = [
            '{',
            '  "type": "FeatureCollection",',
            '  "features": [',
            f'{features_str}',
            '  ]',
            '}'
        ]
        geojson_str = '\n'.join(geojson)

        output_path = os.path.join(self.output_folder, os.path.basename(json_path).replace('.json', '.geojson'))
        with open(output_path, "w") as f:
            f.write(geojson_str)

    def convert_all(self):
        for file in os.listdir(self.input_folder):
            if file.endswith(".json"):
                self.convert_file(os.path.join(self.input_folder, file))


# Usage
input_folder = r"C:\Users\User\Downloads\saad_test"
output_folder = r"C:\Users\User\Downloads\saad_test\geojsons"
metadata_csv_path = r"C:\Users\User\Downloads\image_details_chunk1 (2).csv"

converter = GeoJSONFolderConverter(
    input_folder=input_folder,
    output_folder=output_folder,
    metadata_csv_path=metadata_csv_path
)

converter.convert_all()
