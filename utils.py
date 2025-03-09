import json
# import geojson
# from PIL import Image
# import piexif
import numpy as np
# from collections import defaultdict
# from shapely.geometry import Point, LineString
# from rtree import index
# from collections import defaultdict
from rtree import index
import numpy as np
# from shapely.geometry import box
from pyproj import CRS
import math
from decimal import Decimal, getcontext
import os 
# import pandas as pd
import ast
import traceback
import re

SQUARE_METER = 4046.856

class PostProcess:
    '''
    The post process will handle detection conversion and output generation.
    '''
    def main(self, json_path, box_size, output_path, data, clean_json_path) -> None:
        
        try:
            corners, gsd, width, height, image_name  = self.read_corners_and_gsd_csv(data, json_path)
            Detection_obj = DetectionProcessor(json_path, gsd, box_size)
            clean_detection = Detection_obj.process_detections(clean_json_path, width, height, image_name, corners)
            geojson_obj = GeoJSONConverter(output_path, corners, width, height)
            count = geojson_obj.convert_to_geojson(clean_detection)
            return gsd, width, height, count
        except Exception as e:
            traceback.print_exc()
            print(f"Error occured in {json_path}: {str(e)}")
            
        
    def read_corners_and_gsd_csv(self, data, json_path):
        try:
            image_name = os.path.basename(json_path)
            image_name = os.path.splitext(image_name)[0]
            row = data[data['image_name'] == image_name]
            if not row.empty:
                coordinates = (row.iloc[0]['corners'])
                coordinates = ast.literal_eval(coordinates)
                gsd = row.iloc[0]['gsd']
                width = int(row.iloc[0]['image_width'])
                height = int(row.iloc[0]['image_height'])
                return coordinates, gsd, width, height , image_name
        except Exception as e:
            print(f"Error reading metadata from {json_path}: {str(e)}")
        return None, None, None, None, None
    
      
class DetectionProcessor:
    '''
    Processes and filters detections based on specific criteria.
    '''
    def __init__(self, input_path, gcd, box_size):
        self.input_path = input_path
        self.gcd = gcd
        self.box_size = box_size
        self.detections = self.load_detections()
    def load_detections(self):
        '''
        Loads detections from a JSON file.
        '''
        with open(self.input_path) as f:
            data = json.load(f)
        return data['detections']
    def calculate_center_and_fixed_bbox(self, detections):
        '''
        Calculates center points and fixes bounding box size for each detection.
        '''
        processed_detections = []
        unprocessed_detections = []
        for det in detections:
            if 'pt' in det['name']:
                box = np.array([det['box']['x1'], det['box']['y1'], det['box']['x2'], det['box']['y2']])
                half_size = (self.box_size) / self.gcd
                center = (box[:2] + box[2:]) / 2
                new_box = np.hstack([center - half_size, center + half_size])
                det['box'] = {
                    'x1': new_box[0],
                    'y1': new_box[1],
                    'x2': new_box[2],
                    'y2': new_box[3]
                }
                processed_detections.append(det)
            else:
                unprocessed_detections.append(det)
        return processed_detections, unprocessed_detections
    def detect_and_merge(self, detections):
        '''
        Merges overlapping detections based on bounding box intersection using iterative merging with rtree.
        '''
        final_detections = []
        rtree_idx = index.Index()
        for i, det in enumerate(detections):
            bbox = (det['box']['x1'], det['box']['y1'], det['box']['x2'], det['box']['y2'])
            rtree_idx.insert(i, bbox)
        used = [False] * len(detections)
        for i, det1 in enumerate(detections):
            if used[i]:
                continue
            merged_box = det1['box']
            used[i] = True
            merged = True
            while merged:
                merged = False
                overlapping_idxs = list(rtree_idx.intersection((merged_box['x1'], merged_box['y1'], merged_box['x2'], merged_box['y2'])))
                for j in overlapping_idxs:
                    if not used[j] and detections[j]['name'] == det1['name']:
                        # Check if boxes overlap
                        if (merged_box['x1'] < detections[j]['box']['x2'] and
                            merged_box['x2'] > detections[j]['box']['x1'] and
                            merged_box['y1'] < detections[j]['box']['y2'] and
                            merged_box['y2'] > detections[j]['box']['y1']):
                            merged_box['x1'] = min(merged_box['x1'], detections[j]['box']['x1'])
                            merged_box['y1'] = min(merged_box['y1'], detections[j]['box']['y1'])
                            merged_box['x2'] = max(merged_box['x2'], detections[j]['box']['x2'])
                            merged_box['y2'] = max(merged_box['y2'], detections[j]['box']['y2'])
                            used[j] = True
                            merged = True
            final_detections.append({
                'name': det1['name'],
                'box': merged_box,
                'confidence': det1.get('confidence', 1.0)
            })
        return final_detections


    def calculate_center(self, detections):
        '''
        Computes the center coordinates of each bounding box and structures the output.
        '''
        processed_detections = []
        for det in detections:
            box = det['box']
            x_center = (box['x1'] + box['x2']) / 2
            y_center = (box['y1'] + box['y2']) / 2
            processed_detections.append({
                'name': det['name'],
                'coordinates': [x_center, y_center]
            })
        return processed_detections
    
    def calculate_image_center(self, corners):
        latitudes = [corner[1] for corner in corners]
        longitudes = [corner[0] for corner in corners]

        center_lat = sum(latitudes) / len(latitudes)
        center_lon = sum(longitudes) / len(longitudes)
        
        return center_lat, center_lon
            
    def process_detections(self, clean_json_path, width, height, image_name, corners):
        '''
        Processes detections and returns cleaned results.
        '''
        processed_detections, unprocessed_detections = self.calculate_center_and_fixed_bbox(self.detections)
        combined_detections = processed_detections + unprocessed_detections
        merged_detections = self.detect_and_merge(combined_detections)
        center_lat, center_lon = self.calculate_image_center(corners)
        center_detection = self.calculate_center(processed_detections)
        
        final_json = {
            "ImageHeight": height,
            "ImageWidth": width,
            "ImagePath": image_name,
            "Image_center": (center_lat, center_lon), 
            "detections": center_detection
        }
        with open(clean_json_path, 'w') as outfile:
            # because in final jsons we only need the pt detections
            json.dump(final_json, outfile, indent=4)
        return {'detections': merged_detections}
    

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
        
        for detection in data['detections']:
            x1, y1 = detection['box']['x1'], detection['box']['y1']
            x2, y2 = detection['box']['x2'], detection['box']['y2']
            center_y = (y1 + y2) / 2
            
            if 'pt' in detection['name']:
                center_x = (x1 + x2) / 2
                lat_str, lon_str = self.interpolate_to_gps(center_x, center_y)
                feature = self._create_point_feature(detection, lon_str, lat_str)
                count += 1
                
            elif 'gp' in detection['name']:
                left_lat, left_lon = self.interpolate_to_gps(x1, center_y)
                right_lat, right_lon = self.interpolate_to_gps(x2, center_y)
                feature = self._create_line_feature(detection, left_lon, left_lat, right_lon, right_lat)
            
            features.append(feature)
        
        per_acre_production = count / (area_in_sq_m / 4046.85642)
        geojson = self._build_geojson(features, area_in_sq_m, per_acre_production)
        
        with open(self.output_path, 'w') as geojson_file:
            geojson_file.write(geojson)
        
        return count 

    def _create_point_feature(self, detection, lon_str, lat_str):
        feature = [
            '    {',
            '      "type": "Feature",',
            '      "geometry": {',
            '        "type": "Point",',
            f'        "coordinates": [{lon_str}, {lat_str}]',
            '      },',
            '      "properties": {',
            f'        "name": "{detection["name"]}",',
            f'        "confidence": {detection["confidence"]},',
            '        "type": "Point"',
            '      }',
            '    }'
        ]
        return '\n'.join(feature)

    def _create_line_feature(self, detection, left_lon, left_lat, right_lon, right_lat):
        feature = [
            '    {',
            '      "type": "Feature",',
            '      "geometry": {',
            '        "type": "LineString",',
            f'        "coordinates": [[{left_lon}, {left_lat}], [{right_lon}, {right_lat}]]',
            '      },',
            '      "properties": {',
            f'        "name": "{detection["name"]}",',
            f'        "confidence": {detection["confidence"]},',
            '        "type": "Line"',
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
    def __init__(self, field_json, gsd, image_width, image_height, label, count):
        # self.field_json = field_json
        self.image_width = image_width
        self.image_height = image_height
        self.gsd = gsd
        self.label = label
        self.count = count
        
        with open(field_json, 'r') as data:
            self.field_json = json.load(data)
        

    def one_snap_analysis(self):
        type_label = 'plant_count'
        label = os.path.splitext(os.path.splitext(os.path.basename(self.label))[0])[0]
        total_crop_area_sq = round((self.image_width * self.gsd) * (self.image_height * self.gsd), 2)
        target_population = round(self.field_json.get('target_stand_per_acre') / SQUARE_METER)
        emerged_population = self.count / total_crop_area_sq  # math.ceil if needed
        emergence_rate = emerged_population / target_population * 100
        yield_loss_plants = target_population - emerged_population
        yield_loss_percentage = yield_loss_plants / target_population * 100
    
        color, plant_count = self.get_status_and_color(emergence_rate)
        analysis_results = {
            "type": type_label,
            "label": label,
            "total_crop_area_sq": round(total_crop_area_sq,2),
            "target_population": round(target_population * total_crop_area_sq,0),
            "emerged_population": self.count,
            "emergence_rate": emergence_rate,
            "yield_loss_plants": round(yield_loss_plants * total_crop_area_sq,0),
            "yield_loss_percentage": yield_loss_percentage,
            "color": color,
            "plant_count": plant_count
        }

        return analysis_results 
            
    
    def get_status_and_color(self,target_achieved):
        if target_achieved > 90:
            return "#006400", "Excellent"  # Dark Green
        elif 70 <= target_achieved <= 90:
            return "#008000", "Good"  # Green
        elif 50 <= target_achieved < 70:
            return "#FFFF00", "Average"  # Yellow
        else:
            return "#FF0000", "Poor"  # Red

     
    
    def generate_field_analysis(self, sum_emerged_pop):
        target_population = self.field_json.get('target_stand_per_acre', 1)
        total_crop_area_acres = self.field_json['polygon']['size']
        emerged_population = (sum(sum_emerged_pop) / len(sum_emerged_pop)) * 4046.856
        emergence_rate = (emerged_population / target_population * 100) if target_population else 0
        yield_loss_plants = target_population - emerged_population
        yield_loss_percentage = (yield_loss_plants / target_population * 100) if target_population else 0
        
        color, plant_count = self.get_status_and_color(emergence_rate)

        field_analysis_data = {
            "label": "summary",
            "type": "plant_count",
            "company": "",
            "field_id": f"Field {self.field_json.get('id', '')}",
            "boundary_acres": round(total_crop_area_acres,2), #Done 
            "crop_type": self.field_json.get('cropName', ""),
            "farm": self.field_json.get('name', ""),
            "plantation_date": self.field_json.get('seeding_date', ""),
            "flight_scan_date": self.field_json.get('flight_scan_date', ""),
            "total_crop_area_acres": round(total_crop_area_acres,2),
            "target_population_per_acre": round(target_population,0),
            "emerged_population_per_acre": round(emerged_population,0),
            "total_emerged_plants": round(emerged_population * total_crop_area_acres,0) ,
            "total_target_plants": round(target_population * total_crop_area_acres,0),
            "emergence_rate": round(emergence_rate,2),
            "yield_loss_plants": round(yield_loss_plants,0),
            "yield_loss_percentage": round(yield_loss_percentage,2),
            "color": color,
            "plant_count": plant_count
        }

        return field_analysis_data

        
        
        
        
        
        
        
        