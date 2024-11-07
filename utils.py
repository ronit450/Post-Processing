import json
import geojson
from PIL import Image
import piexif
import numpy as np
from collections import defaultdict
import shapefile
from shapely.geometry import Point, LineString
from rtree import index
from collections import defaultdict
from rtree import index
import numpy as np
from shapely.geometry import box


class PostProcess:
    '''
    The post process will handle detection conversion and output generation.
    '''
    def __init__(self, image_path, json_path, box_size, overlap_threshold, output_path, target_classes=['pt']) -> None:
        self.target_classes = target_classes
        corners, gsd = self.read_corners(image_path)
        Detection_obj = DetectionProcessor(json_path, gsd, box_size, overlap_threshold, target_classes)
        clean_detection = Detection_obj.process_detections()
        geojson_obj = GeoSHPConverter(output_path, image_path, corners)
        geojson_obj.convert_to_shp(clean_detection)

    def read_corners(self, image_path):
        '''
        Reads the corner coordinates and ground sample distance (GSD) from image metadata.
        '''
        try:
            exif_dict = piexif.load(image_path)
            user_comment = exif_dict["Exif"].get(piexif.ExifIFD.UserComment)
            if user_comment and user_comment.startswith(b"XMP\x00"):
                json_data = user_comment[4:].decode('utf-8')
                metadata = json.loads(json_data)
                return metadata.get("corner_coordinates"), metadata.get("gsd")
        except Exception as e:
            print(f"Error reading metadata from {image_path}: {str(e)}")
        return None, None

class DetectionProcessor:
    '''
    Processes and filters detections based on specific criteria.
    '''
    def __init__(self, input_path, gcd, box_size, overlap_threshold, target_classes):
        self.input_path = input_path
        self.gcd = gcd
        self.box_size = box_size
        self.overlap_threshold = overlap_threshold
        self.target_classes = target_classes
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
            if det['name'] in self.target_classes:
                box = np.array(list(det['box'].values()))
                half_size = (self.box_size / 100) / self.gcd  
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
        rtree_idx = index.Index()  # Initialize the R-tree index for fast spatial queries

        # Add each detection to the rtree index
        for i, det in enumerate(detections):
            bbox = (det['box']['x1'], det['box']['y1'], det['box']['x2'], det['box']['y2'])
            rtree_idx.insert(i, bbox)

        used = [False] * len(detections)

        for i, det1 in enumerate(detections):
            if used[i]:
                continue

            # Initialize the merged bounding box with the current detection's box
            merged_box = det1['box']
            used[i] = True
            merged = True

            # Iteratively merge until no more overlaps are found
            while merged:
                merged = False
                overlapping_idxs = list(rtree_idx.intersection((merged_box['x1'], merged_box['y1'], merged_box['x2'], merged_box['y2'])))

                for j in overlapping_idxs:
                    if not used[j] and detections[j]['name'] == det1['name']:  # Only merge if the class is the same
                        # Check if boxes overlap
                        if (merged_box['x1'] < detections[j]['box']['x2'] and 
                            merged_box['x2'] > detections[j]['box']['x1'] and
                            merged_box['y1'] < detections[j]['box']['y2'] and 
                            merged_box['y2'] > detections[j]['box']['y1']):
                            
                            # Update merged_box to encompass both boxes
                            merged_box['x1'] = min(merged_box['x1'], detections[j]['box']['x1'])
                            merged_box['y1'] = min(merged_box['y1'], detections[j]['box']['y1'])
                            merged_box['x2'] = max(merged_box['x2'], detections[j]['box']['x2'])
                            merged_box['y2'] = max(merged_box['y2'], detections[j]['box']['y2'])
                            used[j] = True  # Mark this detection as used
                            merged = True  # Set flag to continue merging

            # Add the fully merged bounding box to the final results
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
        # Separate detections into processed (fixed box size for target classes) and unprocessed
        processed_detections, unprocessed_detections = self.calculate_center_and_fixed_bbox(self.detections)
        
        # Combine processed and unprocessed detections and merge them
        combined_detections = processed_detections + unprocessed_detections
        merged_detections = self.detect_and_merge(combined_detections)
        
        # return {'detections': merged_detections}
        return {'detections': self.detections}

class GeoSHPConverter:
    '''
    Converts detection data to shapefiles with geospatial coordinates.
    '''
    def __init__(self, output_path, image_path, corners):
        self.output_path = output_path
        self.top_left = (corners[1][1], corners[1][0])
        self.bottom_right = (corners[3][1], corners[3][0])
        with Image.open(image_path) as img:
            self.image_width, self.image_height = img.size
        self.gps_corners = {
            "top_left": corners[0],
            "top_right": corners[1],
            "bottom_right": corners[2],
            "bottom_left": corners[3]
        }

    def rotate_point(x, y, angle, image_width, image_height):
        '''
        Rotates a point around the center of the image by a given angle.
        '''
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

    def angle_calculator(self, corners):
        '''
        Calculates the rotation angle from the corner coordinates.
        '''
        edge1 = [corners[1][0] - corners[0][0], corners[1][1] - corners[0][1]]
        edge2 = [corners[3][0] - corners[0][0], corners[3][1] - corners[0][1]]

        # Calculate lengths
        len1 = np.sqrt(edge1[0]**2 + edge1[1]**2)
        len2 = np.sqrt(edge2[0]**2 + edge2[1]**2)

        # Use the longer edge for angle calculation
        if len1 > len2:
            dx, dy = edge1
        else:
            dx, dy = edge2

        angle = np.arctan2(dy, dx)
        print(f"Angle of this image is, {angle}")
        return angle

    def interpolate_to_gps(self, x, y):
        '''
        Interpolates image pixel coordinates to GPS coordinates.
        '''
        norm_x = x / self.image_width
        norm_y = y / self.image_height

        # Bilinear interpolation for longitude and latitude
        lon = (
            self.gps_corners["top_left"][0] * (1 - norm_x) * (1 - norm_y) +
            self.gps_corners["top_right"][0] * norm_x * (1 - norm_y) +
            self.gps_corners["bottom_right"][0] * norm_x * norm_y +
            self.gps_corners["bottom_left"][0] * (1 - norm_x) * norm_y
        )

        lat = (
            self.gps_corners["top_left"][1] * (1 - norm_x) * (1 - norm_y) +
            self.gps_corners["top_right"][1] * norm_x * (1 - norm_y) +
            self.gps_corners["bottom_right"][1] * norm_x * norm_y +
            self.gps_corners["bottom_left"][1] * (1 - norm_x) * norm_y
        )

        return lat, lon

    def convert_to_shp(self, data):
        '''
        Converts detection data to a shapefile with bounding box polygons.
        '''
        shp_writer = shapefile.Writer(self.output_path, shapefile.POLYGON)
        shp_writer.field('Name', 'C')
        shp_writer.field('Confidence', 'F', decimal=2)

        for detection in data['detections']:
            x1, y1 = detection['box']['x1'], detection['box']['y1']
            x2, y2 = detection['box']['x2'], detection['box']['y2']
            
            # Convert each corner of the bounding box to GPS coordinates
            top_left = self.interpolate_to_gps(x1, y1)
            top_right = self.interpolate_to_gps(x2, y1)
            bottom_right = self.interpolate_to_gps(x2, y2)
            bottom_left = self.interpolate_to_gps(x1, y2)

            # Define the polygon as a closed rectangle
            shp_writer.poly([[
                [top_left[1], top_left[0]],      # (lon, lat) for top-left
                [top_right[1], top_right[0]],    # (lon, lat) for top-right
                [bottom_right[1], bottom_right[0]],  # (lon, lat) for bottom-right
                [bottom_left[1], bottom_left[0]],    # (lon, lat) for bottom-left
                [top_left[1], top_left[0]]       # Closing the polygon back to top-left
            ]])

            # Record the name and confidence for each bounding box
            shp_writer.record(detection['name'], detection['confidence'])

        shp_writer.close()
