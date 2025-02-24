import json
import geojson
from PIL import Image
import piexif
import numpy as np
from collections import defaultdict
from shapely.geometry import Point, LineString
from rtree import index
from collections import defaultdict
from rtree import index
import numpy as np
from shapely.geometry import box
from pyproj import CRS
import math
from decimal import Decimal, getcontext
import os 
import pandas as pd
import ast



class PostProcess:
    '''
    The post process will handle detection conversion and output generation.
    '''
    def main(self, json_path, box_size, output_path, data) -> None:
        
        try:
            corners, gsd, width, height  = self.read_corners_and_gsd_csv(data, json_path)
            Detection_obj = DetectionProcessor(json_path, gsd, box_size)
            clean_detection = Detection_obj.process_detections()
            geojson_obj = GeoJSONConverter(output_path, corners, width, height)
            geojson_obj.convert_to_geojson(clean_detection)
            return clean_detection, gsd
        except Exception as e:
            print(f"Error occured in {json_path}: {str(e)}")
        
    def read_corners_and_gsd_csv(self, data, json_path):
        try:
            image_name = os.path.basename(json_path)
            image_name = image_name.replace('.json', '.JPG')
            row = data[data['image_name'] == image_name]
            if not row.empty:
                coordinates = (row.iloc[0]['corners'])
                coordinates = ast.literal_eval(coordinates)
                gsd = row.iloc[0]['gsd']
                width = int(row.iloc[0]['image_width'])
                height = int(row.iloc[0]['image_height'])
                return coordinates, gsd, width, height
        except Exception as e:
            print(f"Error reading metadata from {json_path}: {str(e)}")
        return None, None, None, None
    
    
   
        
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
    def process_detections(self):
        '''
        Processes detections and returns cleaned results.
        '''
        processed_detections, unprocessed_detections = self.calculate_center_and_fixed_bbox(self.detections)
        combined_detections = processed_detections + unprocessed_detections
        merged_detections = self.detect_and_merge(combined_detections)
    
        # with open(r'C:\Users\User\Downloads\pp\new.json', 'w') as outfile:
        #     json.dump({'detections': merged_detections}, outfile, indent=4)
        return {'detections': merged_detections}
    



class GeoJSONConverter:
    '''
    Converts detection data to GeoJSON with geospatial coordinates.
    '''
    def __init__(self, output_path, corners, image_height, image_width):
        self.output_path = output_path
        self.image_height = image_height
        self.image_width = image_width
        getcontext().prec = 18
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
        
        return f"{lat:.16f}", f"{lon:.16f}"
    

    def convert_to_geojson(self, data):
        '''
        Converts detection data to GeoJSON with 16 decimal places precision.
        '''
        features = []
        count = 0
        
        for detection in data['detections']:
            x1, y1 = detection['box']['x1'], detection['box']['y1']
            x2, y2 = detection['box']['x2'], detection['box']['y2']
            center_y = (y1 + y2) / 2
            
            if 'pt' in detection['name']:
                center_x = (x1 + x2) / 2
                center_gps = self.interpolate_to_gps(center_x, center_y)
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [float(center_gps[1]), float(center_gps[0])]  
                    },
                    "properties": {
                        "name": detection['name'],
                        "confidence": detection['confidence'],
                        "type": "Point"
                    }
                }
                features.append(feature)
                count += 1
            elif 'gp' in detection['name']:
                left_gps = self.interpolate_to_gps(x1, center_y)
                right_gps = self.interpolate_to_gps(x2, center_y)
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[float(left_gps[1]), float(left_gps[0])], [float(right_gps[1]), float(right_gps[0])]]
                    },
                    "properties": {
                        "name": detection['name'],
                        "confidence": detection['confidence'],
                        "type": "Line"
                    }
                }
                features.append(feature)

        area_in_sq_m = round(self.image_width * self.image_height, 3)
        geojson_data = {
            "type": "FeatureCollection",
            "properties": {
                "Area": area_in_sq_m,
                "Per Acre Production": count / (area_in_sq_m / 4046.85642),
            },
            "features": features
        }
        
        with open(self.output_path, 'w') as geojson_file:
            json.dump(geojson_data, geojson_file, indent=2)

