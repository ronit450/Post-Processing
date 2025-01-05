import json
import os
import time
import concurrent.futures
import multiprocessing
from utils import *
import traceback
from canopy import * 

class Cleaner:
    def __init__(self, img_folder_path, json_folder_path, detect_output_folder, vari_segment_out, box_size, max_reference_area, vari_bool) -> None:
        self.img_folder_path = img_folder_path
        self.json_folder_path = json_folder_path
        self.detect_out = detect_output_folder
        self.vari_segment_out = vari_segment_out
        self.max_reference_area = max_reference_area
        self.vari_bool = vari_bool
        self.box_size = box_size
        self.post_obj = PostProcess()
        
        
    def process_file(self, file):
        """
        Processes a single file, converting it to a shapefile.
        """
        try:
            if file.endswith('.JPG') or file.endswith('.tif'):
                start_time = time.time()
                image_file = os.path.join(self.img_folder_path, file)
                json_path = os.path.join(self.json_folder_path, file.replace('.JPG', '.json'))
                shp_output_base = os.path.join(self.detect_out, f"{os.path.splitext(file)[0]}.geojson")
                vari_out_geo = os.path.join(self.vari_segment_out, f"{os.path.splitext(file)[0]}.json")
                # Post Processing will output the detected geojson andhere will also provide clean detection which will be sent to VARI
                clean_detection, gsd = self.post_obj.main(
                    image_path=image_file,
                    json_path=json_path,
                    box_size=self.box_size,
                    output_path=shp_output_base
                )
                print(gsd)
                with open(r"C:\Users\User\Downloads\pp\final_detections.json", "w") as f:
                    json.dump(clean_detection, f, indent=4)
                if self.vari_bool:
                    canopy_obj = VegetationSegmentation(gsd=gsd, max_reference_area= self.max_reference_area)
                    canopy_obj.main(image_file, vari_out_geo,  clean_detection)
                
                
                
                end_time = time.time()
                time_taken = end_time - start_time
                print(f"Processed {file} in {time_taken:.2f} seconds")
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            print(traceback.format_exc())
    def process(self):
        """
        Processes all files in the specified folder using a ThreadPoolExecutor.
        """
        files = [f for f in os.listdir(self.img_folder_path) if f.endswith('.JPG') or f.endswith('.tif')]
        for file in files:
            print(file)
            self.process_file(file)


if __name__ == "__main__":
    box_size_cm = 5
    max_reference_area  = 0.12 #This is in meters
    images_folder = r"C:\Users\User\Downloads\pp\imahes"
    json_folder = r"C:\Users\User\Downloads\pp\json"
    vari_segment_out = r"C:\Users\User\Downloads\pp\out_vari"
    post_detection_out =  r"C:\Users\User\Downloads\pp\out"
    
    os.makedirs(post_detection_out, exist_ok=True)
    os.makedirs(vari_segment_out, exist_ok=True)
    
    vari_bool = False # This is kept ke agar baad men VARI nh chahiye 
    processor = Cleaner(
        images_folder, 
        json_folder, 
        post_detection_out, 
        vari_segment_out, 
        box_size_cm,
        max_reference_area, 
        vari_bool 
    )
    processor.process()