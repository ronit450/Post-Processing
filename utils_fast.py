import json
import torch
import numpy as np
from collections import defaultdict
import shapefile
from PIL import Image
import piexif

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
        self.detections, self.boxes, self.classes = self.load_detections()

    def load_detections(self):
        with open(self.input_path) as f:
            data = json.load(f)
        detections = data['detections']
        boxes = torch.tensor([[d['box']['x1'], d['box']['y1'], d['box']['x2'], d['box']['y2']] for d in detections], device='cuda')
        classes = [d['name'] for d in detections]
        return detections, boxes, classes

    def calculate_center_and_fixed_bbox(self):
        half_size = (self.box_size / 100) / self.gcd  
        centers = (self.boxes[:, :2] + self.boxes[:, 2:]) / 2
        new_boxes = torch.cat([centers - half_size, centers + half_size], dim=1)

        processed_detections = [
            dict(self.detections[i], box={
                'x1': new_boxes[i, 0].item(), 'y1': new_boxes[i, 1].item(),
                'x2': new_boxes[i, 2].item(), 'y2': new_boxes[i, 3].item()
            })
            for i in range(len(self.detections)) if self.classes[i] in self.target_classes
        ]

        unprocessed_detections = [
            self.detections[i] for i in range(len(self.detections)) if self.classes[i] not in self.target_classes
        ]

        return processed_detections, unprocessed_detections


    def calculate_iou_tensors(self, box1, box2):
        x1_inter = torch.max(box1[:, 0].unsqueeze(1), box2[:, 0])
        y1_inter = torch.max(box1[:, 1].unsqueeze(1), box2[:, 1])
        x2_inter = torch.min(box1[:, 2].unsqueeze(1), box2[:, 2])
        y2_inter = torch.min(box1[:, 3].unsqueeze(1), box2[:, 3])
        inter_area = torch.clamp(x2_inter - x1_inter, min=0) * torch.clamp(y2_inter - y1_inter, min=0)
        area_box1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area_box2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        union_area = area_box1.unsqueeze(1) + area_box2 - inter_area
        return inter_area / torch.clamp(union_area, min=1e-5)

    def detect_and_merge(self, processed_detections):
        detections_by_class = defaultdict(list)
        for i, detection in enumerate(processed_detections):
            detections_by_class[detection['name']].append((i, detection))

        final_detections = []
        for class_name, class_detections in detections_by_class.items():
            if class_name not in self.target_classes:
                final_detections.extend([det for _, det in class_detections])
                continue

            idxs = [idx for idx, _ in class_detections]
            class_boxes = self.boxes[idxs]
            to_keep = torch.ones(len(class_boxes), dtype=torch.bool, device='cuda')
            for i in range(len(class_boxes)):
                if not to_keep[i]:
                    continue
                ious = self.calculate_iou_tensors(class_boxes[i].unsqueeze(0), class_boxes)
                overlapping_idxs = (ious > self.overlap_threshold).nonzero(as_tuple=True)[1]
                best_idx = overlapping_idxs[torch.argmax(ious[0, overlapping_idxs])]
                to_keep[overlapping_idxs] = False
                to_keep[best_idx] = True

            final_detections.extend([class_detections[idx][1] for idx in range(len(class_boxes)) if to_keep[idx]])

        return final_detections

    def process_detections(self):
        processed_detections, unprocessed_detections = self.calculate_center_and_fixed_bbox()
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

    def interpolate_to_gps(self, x, y):
        norm_x = x / self.image_width
        norm_y = y / self.image_height
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

