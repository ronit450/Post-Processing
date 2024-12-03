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
class PostProcess:
    '''
    The post process will handle detection conversion and output generation.
    '''
    def __init__(self, image_path, json_path, box_size, output_path) -> None:
        corners, gsd,  image_width_aiman, image_height_aiman= self.read_corners_and_gsd_from_exif(image_path)
        self.new_height_aiman = image_height_aiman
        self.new_width_aiman = image_width_aiman
        Detection_obj = DetectionProcessor(json_path, gsd, box_size)
        clean_detection = Detection_obj.process_detections()
        geojson_obj = GeoJSONConverter(output_path, image_path, corners, self.new_height_aiman, self.new_width_aiman)
        geojson_obj.convert_to_geojson(clean_detection)
    def read_corners_and_gsd_from_exif(self, image_path):
        try:
            exif_dict = piexif.load(image_path)
            user_comment = exif_dict["Exif"].get(piexif.ExifIFD.UserComment)
            if user_comment and user_comment.startswith(b"XMP\x00"):
                json_data = user_comment[4:].decode('utf-8')
                metadata = json.loads(json_data)
                return metadata.get("Corner_Coordinates"), metadata.get("GSD"), metadata.get("ImageWidth_Meter"), metadata.get("ImageHeigth_Meter")
        except Exception as e:
            print(f"Error reading metadata from {image_path}: {str(e)}")
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
        return json.load(data['detections'])

    def calculate_center_and_fixed_bbox(self, detections):
        '''
        Calculates center points and creates a new bounding box with specified radius in cm.
        self.box_size: radius in centimeters
        self.gcd: scale in meters
        '''
        processed_detections = []
        unprocessed_detections = []
        for det in detections:
            if 'pt' in det['name']:
                box = np.array([det['box']['x1'], det['box']['y1'], 
                            det['box']['x2'], det['box']['y2']])
                
                # Convert box_size from cm to coordinate units
                # First convert cm to meters (/100) then divide by GCD to match coordinate scale
                radius = (self.box_size / 100) / self.gcd
                
                # Calculate center of original box
                center = (box[:2] + box[2:]) / 2
                
                # Create new box extending radius from center
                new_box = np.array([
                    center[0] - radius,  # x1
                    center[1] - radius,  # y1
                    center[0] + radius,  # x2
                    center[1] + radius   # y2
                ])
                
                det['box'] = {
                    'x1': new_box[0],
                    'y1': new_box[1],
                    'x2': new_box[2],
                    'y2': new_box[3]
                }
                processed_detections.append(det)
            else:
                unprocessed_detections.append(det)
        
        with open(r'C:\Users\User\Downloads\pp\new.json', 'w') as outfile:
            json.dump({'detections': detections}, outfile, indent=4)
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
        # print(f"merged_det = {merged_detections}") ab chalae
        return {'detections': merged_detections}
        # return {'detections': combined_detections}
class GeoJSONConverter:
    '''
    Converts detection data to GeoJSON with geospatial coordinates.
    '''
    def __init__(self, output_path, image_path, corners, image_height_a, image_width_a):
        self.output_path = output_path
        self.image_height_aiman = image_height_a
        self.image_weight_aiman = image_width_a
        
        # Store corners in a more explicit format
        self.gps_corners = {
            "top_left": (corners[0][0], corners[0][1]),     # (lon, lat)
            "top_right": (corners[1][0], corners[1][1]),    # (lon, lat)
            "bottom_right": (corners[2][0], corners[2][1]), # (lon, lat)
            "bottom_left": (corners[3][0], corners[3][1])   # (lon, lat)
        }
        
        # Load image dimensions
        with Image.open(image_path) as img:
            self.image_width, self.image_height = img.size
            
    def interpolate_to_gps(self, x, y):
        """
        Convert pixel coordinates to GPS coordinates using bilinear interpolation.
        
        Args:
            x (float): x-coordinate in pixel space
            y (float): y-coordinate in pixel space
        
        Returns:
            tuple: (latitude, longitude) coordinates
        """
        # Ensure coordinates are within image bounds
        x = max(0, min(x, self.image_width))
        y = max(0, min(y, self.image_height))
        
        # Calculate normalized coordinates [0,1]
        u = x / self.image_width
        v = y / self.image_height
        
        # Calculate weights for bilinear interpolation
        w00 = (1 - u) * (1 - v)  # top-left weight
        w10 = u * (1 - v)        # top-right weight
        w11 = u * v              # bottom-right weight
        w01 = (1 - u) * v        # bottom-left weight
        
        # Interpolate longitude
        lon = (self.gps_corners["top_left"][0] * w00 +
            self.gps_corners["top_right"][0] * w10 +
            self.gps_corners["bottom_right"][0] * w11 +
            self.gps_corners["bottom_left"][0] * w01)
        
        # Interpolate latitude
        lat = (self.gps_corners["top_left"][1] * w00 +
            self.gps_corners["top_right"][1] * w10 +
            self.gps_corners["bottom_right"][1] * w11 +
            self.gps_corners["bottom_left"][1] * w01)
        
        return (lat, lon)


    def convert_to_geojson(self, data):
        """
        Converts detection data to GeoJSON format with points and lines.
        
        Args:
            data (dict): Dictionary containing detection data
            
        Returns:
            None: Writes GeoJSON file to self.output_path
        """
        features = []
        count = 0
        
        for detection in data['detections']:
            # Calculate center of bounding box
            center_x = (detection['box']['x1'] + detection['box']['x2']) / 2
            center_y = (detection['box']['y1'] + detection['box']['y2']) / 2
            
            if 'pt' in detection['name']:
                # Get GPS coordinates for center point
                lat, lon = self.interpolate_to_gps(center_x, center_y)
                
                # Create point feature
                feature = geojson.Feature(
                    geometry=geojson.Point((lon, lat)),  # GeoJSON uses (lon, lat) order
                    properties={
                        "name": detection['name'],
                        "confidence": detection['confidence'],
                        "type": "Point",
                        "pixel_x": center_x,  # Add original pixel coordinates for debugging
                        "pixel_y": center_y
                    }
                )
                features.append(feature)
                count += 1
                
            elif 'gp' in detection['name']:
                # Get GPS coordinates for line endpoints
                x1, x2 = detection['box']['x1'], detection['box']['x2']
                left_lat, left_lon = self.interpolate_to_gps(x1, center_y)
                right_lat, right_lon = self.interpolate_to_gps(x2, center_y)
                
                # Create line feature
                feature = geojson.Feature(
                    geometry=geojson.LineString([
                        (left_lon, left_lat),
                        (right_lon, right_lat)
                    ]),
                    properties={
                        "name": detection['name'],
                        "confidence": detection['confidence'],
                        "type": "Line",
                        "pixel_start": (x1, center_y),  # Add original pixel coordinates
                        "pixel_end": (x2, center_y)
                    }
                )
                features.append(feature)

        # Calculate area and per acre production
        area_in_sq_m = round(self.image_weight_aiman * self.image_height_aiman, 3)
        per_acre_production = count / (area_in_sq_m / 4046.85642)  # Convert to acres
        
        # Create final GeoJSON structure
        geojson_data = {
            "type": "FeatureCollection",
            "properties": {
                "Area": area_in_sq_m,
                "Per_Acre_Production": round(per_acre_production, 2),
                "Total_Points": count
            },
            "features": features
        }
        
        # Write to GeoJSON file
        with open(self.output_path, 'w') as geojson_file:
            geojson.dump(geojson_data, geojson_file, indent=2)