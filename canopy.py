import cv2
import numpy as np
import json
import os
from PIL import Image
from skimage.segmentation import felzenszwalb
from skimage.morphology import binary_closing, remove_small_objects, disk
from skimage.filters import gaussian

class VegetationSegmentation:
    def __init__(self, gsd, max_reference_area):
        self.gsd = gsd
        self.pixel_area = gsd * gsd
        self.max_reference_area = max_reference_area
        
    def expand_bbox(self, bbox: np.ndarray, expansion_factor: float = 0.15) -> np.ndarray:
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        x_expand = width * expansion_factor
        y_expand = height * expansion_factor
        
        return np.array([
            max(0, bbox[0] - x_expand),
            max(0, bbox[1] - y_expand),
            bbox[2] + x_expand,
            bbox[3] + y_expand
        ]).astype(int)

    def get_vegetation_mask(self, image):
        """Get initial vegetation mask using multiple indices"""
        # Convert to float and normalize
        img_float = image.astype(np.float32) / 255.0
        
        # Extract channels
        red = img_float[:,:,0]
        green = img_float[:,:,1]
        blue = img_float[:,:,2]
        
        # Calculate ExG (Excess Green Index)
        exg = 2 * green - red - blue
        
        # Calculate VARI (Visible Atmospherically Resistant Index)
        denominator = green + red - blue
        denominator[denominator == 0] = 1e-7
        vari = (green - red) / denominator
        
        # Convert to HSV for additional green detection
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Create masks
        exg_mask = (exg > 0.1).astype(np.uint8) * 255
        vari_mask = (vari > 0.1).astype(np.uint8) * 255
        hsv_mask = cv2.inRange(hsv, (35, 20, 20), (85, 255, 255))
        
        # Combine masks
        combined_mask = cv2.bitwise_or(
            cv2.bitwise_or(exg_mask, vari_mask),
            hsv_mask
        )
        
        return combined_mask

    def refine_segments(self, image, mask):
        """Refine segmentation using Felzenszwalb"""
        # Apply Felzenszwalb segmentation
        segments = felzenszwalb(image, scale=100, sigma=0.5, min_size=50)
        
        refined_mask = np.zeros_like(mask)
        
        # Process each segment
        unique_segments = np.unique(segments)
        for segment_id in unique_segments:
            segment_mask = segments == segment_id
            overlap = np.sum(mask[segment_mask] > 0) / np.sum(segment_mask)
            if overlap > 0.3:  # If 30% overlap with vegetation
                refined_mask[segment_mask] = 255
        
        return refined_mask

    def clean_mask(self, mask):
        """Clean the segmentation mask"""
        # Apply morphological closing
        kernel = np.ones((5,5), np.uint8)
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Remove small objects
        cleaned = remove_small_objects(cleaned.astype(bool), min_size=100).astype(np.uint8) * 255
        
        return cleaned

    def get_vegetation_class(self, area: float) -> int:
        relative_size = (area / self.max_reference_area) * 100
        if relative_size <= 25:
            return 1
        elif relative_size <= 50:
            return 2
        elif relative_size <= 75:
            return 3
        else:
            return 4

    def process_bbox(self, image_path: str, bbox: np.ndarray) -> tuple:
        # Read image
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img)
        
        # Ensure bbox coordinates are integers
        bbox = bbox.astype(int)
        x1, y1, x2, y2 = bbox
        
        # Crop image to bbox
        cropped = img_np[y1:y2, x1:x2]
        
        # Get vegetation mask
        veg_mask = self.get_vegetation_mask(cropped)
        
        # Refine segmentation
        refined_mask = self.refine_segments(cropped, veg_mask)
        
        # Clean mask
        final_mask = self.clean_mask(refined_mask)
        
        return final_mask, cropped

    def simplify_contour(self, contour, epsilon_factor=0.005):
        """Simplify contour while preserving shape"""
        perimeter = cv2.arcLength(contour, True)
        epsilon = epsilon_factor * perimeter
        simplified = cv2.approxPolyDP(contour, epsilon, True)
        return simplified

    def main(self, image_path, output_path, detections):
        all_segments = []
        
        for bbox_data in detections['detections']:
            # Get bbox coordinates
            bbox = np.array([
                float(bbox_data["box"]["x1"]),
                float(bbox_data["box"]["y1"]),
                float(bbox_data["box"]["x2"]),
                float(bbox_data["box"]["y2"])
            ])
            
            # Expand bbox slightly
            expanded_box = self.expand_bbox(bbox)
            
            try:
                # Process the bbox
                mask, cropped = self.process_bbox(image_path, expanded_box)
                
                # Find contours
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    # Filter small contours
                    if cv2.contourArea(contour) < 100:
                        continue
                    
                    # Simplify contour
                    simplified = self.simplify_contour(contour)
                    
                    # Calculate area
                    area = cv2.contourArea(simplified) * self.pixel_area
                    
                    # Get vegetation class
                    veg_class = self.get_vegetation_class(area)
                    
                    # Adjust coordinates back to original image space
                    adjusted_contour = simplified.squeeze() + [expanded_box[0], expanded_box[1]]
                    points = adjusted_contour.tolist()
                    
                    if isinstance(points, list) and len(points) > 2:
                        if not isinstance(points[0], list):
                            points = [points]
                        
                        segment_data = {
                            "label": "pt",
                            "points": [[float(pt[0]), float(pt[1])] for pt in points],
                            "group_id": None,
                            "description": None,
                            "vegetation_class": veg_class,
                            "area_sq_meters": float(area),
                            "relative_size_percent": float((area / self.max_reference_area) * 100)
                        }
                        all_segments.append(segment_data)
                        
            except Exception as e:
                print(f"Error processing bbox {bbox}: {str(e)}")
                continue
        
        # Prepare output data
        data = {
            "version": "0.3.3",
            "flags": {},
            "shapes": all_segments,
            "imagePath": os.path.basename(image_path),
            "imageData": None,
            "metadata": {
                "gsd": self.gsd,
                "max_reference_area": self.max_reference_area,
                "class_definitions": {
                    "class_1": "0-25% of reference area",
                    "class_2": "25-50% of reference area",
                    "class_3": "50-75% of reference area",
                    "class_4": "75-100+% of reference area"
                }
            }
        }
        
        # Save output
        with open(output_path, "w") as f:
            json.dump(data, f, indent=4)

