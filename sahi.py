import os
import cv2
import json
import numpy as np
from pathlib import Path
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.predict import predict
from sahi.utils.file import list_files
import piexif
import numpy as np
import json
from skimage import measure
import ast
import pandas as pd

class SahiDetect:
    def __init__(self, image_folder, model_path, phsz_width, phsz_height, imgsz, output_dir, task, csv_path):
        self.image_folder = image_folder
        self.model_path= model_path
        self.phsz_width = phsz_width
        self.phsz_height = phsz_height
        self.imgsz  = imgsz
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.task = task
        self.data = pd.read_csv(csv_path)
    
    def read_gsd_csv(self, json_path):
        try:
            image_name = os.path.basename(json_path)
            print(image_name)
            row = self.data[self.data['image_name'] == image_name]
            if not row.empty:
                coordinates = (row.iloc[0]['corners'])
                coordinates = ast.literal_eval(coordinates)
                gsd = row.iloc[0]['gsd']
                width = int(row.iloc[0]['image_width'])
                height = int(row.iloc[0]['image_height'])
                return  gsd, width, height 
        except Exception as e:
            print(f"Error reading metadata from {json_path}: {str(e)}")
        return None, None, None
    
    def model_loading(self):
        self.detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=self.model_path,
        confidence_threshold=0.2,
        device="cuda:0", 
        # image_size= 3000
        )
    
    def read_corners_and_gsd_from_exif(self, image_path):
        try:
            exif_dict = piexif.load(image_path)
            user_comment = exif_dict["Exif"].get(piexif.ExifIFD.UserComment)
            if user_comment and user_comment.startswith(b"XMP\x00"):
                json_data = user_comment[4:].decode('utf-8')
                metadata = json.loads(json_data)
                return  metadata.get("GSD"), metadata.get("ImageWidth_Meter"), metadata.get("ImageHeigth_Meter")
                # return  metadata.get("GSD"), 5280, 3956

        except Exception as e:
            print(f"Error reading metadata from {image_path}: {str(e)}")
        return None, None, None
    
    
    def calculate_tile_size(self, gsd, img_width, img_height):
        img_height, img_width = img_height, img_width
        tile_width_px = int(self.phsz_width / gsd)
        tile_height_px = int(self.phsz_height / gsd)
        tile_width_px = min(tile_width_px, img_width)
        tile_height_px = min(tile_height_px, img_height)
        tile_width_px = (tile_width_px // 32) * 32
        tile_height_px = (tile_height_px // 32) * 32
        tile_width_px = max(tile_width_px, 32)
        tile_height_px = max(tile_height_px, 32)
        return tile_width_px, tile_height_px
            
    def process_images(self):
        self.model_loading()
        image_paths = list_files(
            self.image_folder, 
            contains=[".jpg", ".jpeg", ".png", ".tif", ".tiff"],)    
        for image_path in image_paths:
            image_filename = os.path.basename(image_path)
            json_path = os.path.join(self.output_dir, f'{os.path.splitext(os.path.basename(image_path))[0]}.json')
            print(f"Processing {image_filename}")
            gsd, image_width, image_height = self.read_gsd_csv(image_path)
            print(gsd, image_width, image_height )
            tile_width, tile_height = self.calculate_tile_size(gsd,  image_width, image_height)
            results = get_sliced_prediction(
            image_path, 
            self.detection_model,
            slice_height= tile_width,
            slice_width=tile_height,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            )
            if self.task == 0:
                json_result = self.convert_sahi_results_OD(results)
            else:
                json_result = self.convert_sahi_segmentation(results)
            
            with open(json_path, 'w') as f:
                json.dump(json_result, f, indent=4)
            
            
            
    
    def convert_sahi_results_OD(self, results):
        json_detections = []
        
        for prediction in results.object_prediction_list:
            x_min = prediction.bbox.minx
            y_min = prediction.bbox.miny
            x_max = prediction.bbox.maxx
            y_max = prediction.bbox.maxy
            
            confidence = prediction.score.value
            class_id = prediction.category.id
            class_name = prediction.category.name
            
            json_detections.append({
                "name": class_name,
                "class": class_id,
                "confidence": confidence,
                "box": {
                    "x1": x_min,
                    "y1": y_min,
                    "x2": x_max,
                    "y2": y_max
                }
            })
        
        json_result = {
            "detections": json_detections
        }
        
        return json_result
    
    def convert_sahi_segmentation(self, results):
        def extract_mask_data(mask_obj):
            if hasattr(mask_obj, 'points'):
                return mask_obj.points
            elif hasattr(mask_obj, 'polygons'):
                return mask_obj.polygons
            elif hasattr(mask_obj, 'segmentation'):
                return mask_obj.segmentation
            elif hasattr(mask_obj, 'mask') and isinstance(mask_obj.mask, np.ndarray):
                contours = measure.find_contours(mask_obj.mask.astype(np.uint8), 0.5)
                polygons = []
                for contour in contours:
                    polygon = []
                    for point in contour:
                        polygon.extend([float(point[1]), float(point[0])])
                    if len(polygon) >= 6:
                        polygons.append(polygon)
                return polygons
            else:
                try:
                    mask_str = str(mask_obj)
                    if '[' in mask_str and ']' in mask_str:
                        import re
                        arrays = re.findall(r'\[.*?\]', mask_str)
                        if arrays:
                            return arrays
                except:
                    pass
                
                return []
        
        json_detections = []
        
        for prediction in results.object_prediction_list:
            confidence = prediction.score.value
            class_id = prediction.category.id
            class_name = prediction.category.name
            
            mask_data = extract_mask_data(prediction.mask)
            
            json_detections.append({
                "name": class_name,
                "class": class_id,
                "confidence": confidence,
                "segmentation": mask_data
            })
        
        json_result = {
            "detections": json_detections
        }
        
        return json_result

                    

# if __name__ == "__main__":
#     sahi_processor = sahiInference(
#         image_folder= r"C:\Users\User\Desktop\Ronit-Projects\Farmevo-Sorted-Projects\Ronit-Farmevo\SAHI-TRIAL\sample-data\momna-test", 
#         model_path= r"C:\Users\User\Downloads\Compressed\new_1600\pt_20250313_080217.pt", 
#         phsz_width=4,
#         phsz_height=3, 
#         imgsz=1600, 
#         task = 0 , 
#         output_dir =r"C:\Users\User\Desktop\Ronit-Projects\Farmevo-Sorted-Projects\Ronit-Farmevo\SAHI-TRIAL\sample-data\momna-test"
#     )
    
#     results = sahi_processor.process_images()       