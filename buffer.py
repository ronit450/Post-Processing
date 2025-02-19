import json
import os
import time
import concurrent.futures
import multiprocessing
from utils import *
import traceback
import pandas as pd
from canopy import * 

class Cleaner:
    def __init__(self, json_folder_path, detect_output_folder, box_size, max_reference_area, csv_path) -> None:
        self.json_folder_path = json_folder_path
        self.detect_out = detect_output_folder
        self.max_reference_area = max_reference_area
        self.box_size = box_size
        self.data = pd.read_csv(csv_path)
        self.post_obj = PostProcess()
        
        
    def process_file(self, file):
        """
        Processes a single file, converting it to a shapefile.
        """
        try:
            start_time = time.time()
            json_path = os.path.join(self.json_folder_path, file)
            shp_output_base = os.path.join(self.detect_out, f"{os.path.splitext(file)[0]}.geojson")
            # Post Processing will output the detected geojson andhere will also provide clean detection which will be sent to VARI
            self.post_obj.main(
                json_path=json_path,
                box_size=self.box_size,
                output_path=shp_output_base, 
                data = self.data
            )
                                
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
        files = [f for f in os.listdir(self.json_folder_path) if f.endswith('.json')]
        for file in files:
            self.process_file(file)


if __name__ == "__main__":
    box_size_cm = 5
    max_reference_area  = 0.12 #This is in meters
    
    json_folder = r"C:\Users\User\Downloads\test_json"
    post_detection_out =  r"C:\Users\User\Downloads\test_json_result"
    csv_path = r"C:\Users\User\Downloads\image_details (2).csv"
    
    os.makedirs(post_detection_out, exist_ok=True)
    
    processor = Cleaner(
        json_folder, 
        post_detection_out, 
        box_size_cm,
        max_reference_area, 
        csv_path
    )
    processor.process()