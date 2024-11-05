import json
import geojson
from PIL import Image
import piexif
import numpy as np
from collections import defaultdict
import shapefile
from shapely.geometry import Point, LineString
from rtree import index

class PostProcess:
    def __init__(self, image_path, json_path, box_size, overlap_threshold, output_path, target_classes=['pt']) -> None:
        self.target_classes = target_classes
        corners, gsd = self.read_corners(image_path)
        Detection_obj = DetectionProcessor(json_path, gsd, box_size, overlap_threshold, target_classes)
        clean_detection = Detection_obj.process_detections()
        geojson_obj = GeoSHPConverter(output_path, image_path, corners)
        geojson_obj.convert_to_shp(clean_detection)

    def read_corners(self, image_path):
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
    def __init__(self, input_path, gcd, box_size, overlap_threshold, target_classes):
        self.input_path = input_path
        self.gcd = gcd
        self.box_size = box_size
        self.overlap_threshold = overlap_threshold
        self.target_classes = target_classes
        self.detections = self.load_detections()

    def load_detections(self):
        with open(self.input_path) as f:
            data = json.load(f)
        return data['detections']

    def calculate_center_and_fixed_bbox(self, detections):
        processed_detections = []
        unprocessed_detections = []
        for det in detections:
            if det['name'] in self.target_classes:
                box = np.array(list(det['box'].values()))
                center = (box[:2] + box[2:]) / 2
                half_size = (self.box_size) * (self.gcd * 100)
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

    def calculate_iou(self, box1, box2):
        x1_inter = np.maximum(box1[:, 0], box2[:, 0])
        y1_inter = np.maximum(box1[:, 1], box2[:, 1])
        x2_inter = np.minimum(box1[:, 2], box2[:, 2])
        y2_inter = np.minimum(box1[:, 3], box2[:, 3])
        inter_area = np.maximum(x2_inter - x1_inter, 0) * np.maximum(y2_inter - y1_inter, 0)
        area_box1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area_box2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        union_area = area_box1 + area_box2 - inter_area
        return inter_area / np.maximum(union_area, 1e-5)

    def detect_and_merge(self, detections):
        detections_by_class = defaultdict(list)
        for det in detections:
            detections_by_class[det['name']].append(det)

        final_detections = []
        for class_name, class_detections in detections_by_class.items():
            bboxes = np.array([list(d['box'].values()) for d in class_detections])
            rtree_idx = index.Index()
            for i, det in enumerate(class_detections):
                bbox = (det['box']['x1'], det['box']['y1'], det['box']['x2'], det['box']['y2'])
                rtree_idx.insert(i, bbox)

            to_keep = np.ones(len(class_detections), dtype=bool)
            for i in range(len(class_detections)):
                if not to_keep[i]:
                    continue
                overlapping_idxs = list(rtree_idx.intersection(bboxes[i]))
                if len(overlapping_idxs) > 1:
                    ious = self.calculate_iou(bboxes[i].reshape(1, -1), bboxes[overlapping_idxs])
                    best_idx = overlapping_idxs[np.argmax(ious)]
                    to_keep[overlapping_idxs] = False
                    to_keep[best_idx] = True

            final_detections.extend([class_detections[i] for i in range(len(class_detections)) if to_keep[i]])
        return final_detections

    def process_detections(self):
        processed_detections, unprocessed_detections = self.calculate_center_and_fixed_bbox(self.detections)
        merged_detections = self.detect_and_merge(processed_detections)
        final_detections = merged_detections + unprocessed_detections
        return {'detections': final_detections}

class GeoSHPConverter:
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
        
        """
        Calculate the correct rotation angle from corners
        """
        # Get the main edges
        
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
        return angle


    def interpolate_to_gps(self, x, y):
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
        shp_writer = shapefile.Writer(self.output_path, shapefile.POINT)
        shp_writer.field('Name', 'C')
        shp_writer.field('Confidence', 'F', decimal=2)
        for detection in data['detections']:
            x1, y1 = detection['box']['x1'], detection['box']['y1']
            x2, y2 = detection['box']['x2'], detection['box']['y2']
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            lat, lon = self.interpolate_to_gps(center_x, center_y)
            shp_writer.point(lon, lat)
            shp_writer.record(detection['name'], detection['confidence'])
        shp_writer.close()
