import numpy as np
from rtree import index
import numpy as np
from decimal import Decimal, getcontext
import os 
import ast
import traceback
import numpy as np
import cv2  
import json 



class GeoJSONConverter:
    '''
    Converts detection data to GeoJSON with geospatial coordinates.
    '''
    def __init__(self, output_path, corners, image_width, image_height):
        self.output_path = output_path
        self.image_height = image_height
        self.image_width = image_width
        getcontext().prec = 48
        self.gps_corners = {
            "top_left": corners[0],
            "top_right": corners[1],
            "bottom_right": corners[2],
            "bottom_left": corners[3]
        }
    
    def interpolate_to_gps(self, x, y):
        '''
        Interpolates image pixel coordinates to GPS coordinates with high precision.
        Returns Decimal values for maximum precision.
        '''
        norm_x = Decimal(x) / Decimal(self.image_width)
        norm_y = Decimal(y) / Decimal(self.image_height)
        
        lon = (
            Decimal(self.gps_corners["top_left"][0]) * (1 - norm_x) * (1 - norm_y) +
            Decimal(self.gps_corners["top_right"][0]) * norm_x * (1 - norm_y) +
            Decimal(self.gps_corners["bottom_right"][0]) * norm_x * norm_y +
            Decimal(self.gps_corners["bottom_left"][0]) * (1 - norm_x) * norm_y
        )
        
        lat = (
            Decimal(self.gps_corners["top_left"][1]) * (1 - norm_x) * (1 - norm_y) +
            Decimal(self.gps_corners["top_right"][1]) * norm_x * (1 - norm_y) +
            Decimal(self.gps_corners["bottom_right"][1]) * norm_x * norm_y +
            Decimal(self.gps_corners["bottom_left"][1]) * (1 - norm_x) * norm_y
        )
        
        return f"{lat:.42f}", f"{lon:.42f}"
    
    def convert_to_geojson(self, data):
        area_in_sq_m = round(self.image_width * self.image_height, 3)
        count = 0
        features = []

        for shape in data.get("shapes", []):
            if shape.get("shape_type") != "point":
                continue  # Only process point annotations

            x, y = shape["points"][0]
            lat_str, lon_str = self.interpolate_to_gps(x, y)
            temp_area = area_in_sq_m  # Area in acres

            # Optionally use shape["label"] or default label
            label = shape.get("label", "point")

            feature = self._create_point_feature({"label": "corty"}, lon_str, lat_str, temp_area)
            features.append(feature)
            count += 1

        per_acre_production = count / (area_in_sq_m / 4046.85642)
        geojson = self._build_geojson(features, area_in_sq_m, per_acre_production)

        with open(self.output_path, 'w') as geojson_file:
            geojson_file.write(geojson)

        return count


    def _create_point_feature(self, detection, lon_str, lat_str, temp_area):
        feature = [
            '    {',
            '      "type": "Feature",',
            '      "geometry": {',
            '        "type": "Point",',
            f'        "coordinates": [{lon_str}, {lat_str}]',
            '      },',
            '      "properties": {',
            f'        "name": "corty",',
            f'        "area_m2": {temp_area},',    
            '        "type": "Point"',
            '      }',
            '    }'
        ]
        return '\n'.join(feature)


    def _build_geojson(self, features, area, per_acre):
        features_str = ',\n'.join(features)
        
        geojson_parts = [
            '{',
            '  "type": "FeatureCollection",',
            '  "properties": {',
            f'    "Area": {area},',
            f'    "Per Acre Production": {per_acre}',
            '  },',
            '  "features": [',
            f'{features_str}',
            '  ]',
            '}'
        ]
        
        return '\n'.join(geojson_parts)






class Analysis:
    def __init__(self, field_json, gsd, image_width, image_height, label, count, corners):
        # self.field_json = field_json
        self.image_width = image_width
        self.image_height = image_height
        self.gsd = gsd
        self.label = label
        self.count = count 
        self.corners = corners
        self.company = "Corteva"
        with open(field_json, 'r') as data:
            self.field_json = json.load(data)
            
        self.thresholds = {
        "Hytech": [(0.00247, 0), (0.00123, 1), (0.00025, 2), (0, 3)],
        "Corteva": [(0.01236, 0), (0.00519, 1), (0.00025, 2), (0, 3)],
        "Bayer": [(0.01236, 0), (0.00618, 1), (0.00123, 2), (0, 3)],
        "Nutrien": [(0.00247, 0), (0.00123, 1), (0.00025, 2), (0, 3)],
    }
    
    
    def summary_class_calculator(self, number):
        if number > 50:
            return 0
        elif number >=21 and number <= 50:
            return 1
        elif number >= 1 and number <= 20:
            return 2
        else:
            return 3
        
    
    def count_detections(self, json_path):
        with open(json_path, 'r') as file:
            data = json.load(file)
    
        detections = data.get("Image_detections", [])
        return len(detections)
    
    

    def assign_class(self, count, company):
        for limit, cls in self.thresholds.get(company, []):
            if count > limit:
                return cls
        return 3
        
        

    def one_snap_analysis(self):
        type_label = 'OffType'
        label = os.path.splitext(os.path.splitext(os.path.basename(self.label))[0])[0]
        total_crop_area_sq = round((self.image_width * self.gsd) * (self.image_height * self.gsd), 2)
        
        
        analysis_results = {
            "type": type_label,
            "label": label,
            "total_crop_area_sq": round(total_crop_area_sq,2),
            "crop_type": "Canola Pre-Scout",
            "offType": self.count, 
            "class": self.assign_class(self.count, self.company),
        }

        return analysis_results , total_crop_area_sq
            
    
    
    def generate_field_analysis(self, total_count, total_image_area, image_count):
        
        total_crop_area_acres = self.field_json['polygon']['size']
        temp_number = (4046.85642/ (image_count * 10))* total_count
        field_analysis_data = {
            "label": "summary",
            "type": "OffType",
            "company": "Corteva",
            "field_id": f"Field {self.field_json.get('id', '')}",
            "seeded_area" : self.field_json.get('polygon', {}).get('size', 0),
            "crop_type": "Canola Pre-Scout",
            "total_scouted_area" : total_image_area,
            "total_scouted_area_1" : image_count * 10,  
            "farm": self.field_json.get('name', ""),
            "offtypes_per_acre": temp_number,  # Convert m^2 to acres
            "seeded_date": self.field_json.get('seeding_date', ""),
            "flight_scan_date": self.field_json.get('flight_scan_date', ""),
            "total_crop_area_acres": round(total_crop_area_acres,2),
            "off_type_Count": total_count,
            "class": self.summary_class_calculator(temp_number),
        }

        return field_analysis_data
