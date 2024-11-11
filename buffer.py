import json
import numpy as np
import os
import time
from utils import *

class Cleaner:
    def __init__(self, img_folder_path, json_folder_path, output_folder, box_size) -> None:
        self.img_folder_path = img_folder_path
        self.json_folder_path = json_folder_path
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)
        self.box_size = box_size

    def process(self):
        for file in os.listdir(self.img_folder_path):
            if file.endswith('.JPG') or file.endswith('.tif'):
                start_time = time.time()
                image_file = os.path.join(self.img_folder_path, file)
                json_file = os.path.splitext(file)[0] + '.json'
                print(f"Processing {json_file}")
                json_path = os.path.join(self.json_folder_path, json_file)
                
                shp_output_base = os.path.join(self.output_folder, os.path.splitext(file)[0])
                
                PostProcess(
                    image_path=image_file, 
                    json_path=json_path, 
                    box_size=self.box_size, 
                    output_path=shp_output_base  
                )
                
                end_time = time.time()
                time_taken = end_time - start_time
                print(f"Processed {file} in {time_taken:.2f} seconds")

if __name__ == "__main__":
    box_size_cm = 5  # Size of potato in cm

    processor = Cleaner(
        r"E:\blackgold\nutfarm2\DCIM\output_test",  # Input folder path
        r"D:\Aiman\25-10-2024_complete_pipeline_test\yolo_soybean\code\jsons", #json folder path
        r"E:\blackgold\nutfarm2\DCIM\output_geojsons",  # Output folder path
        box_size_cm,
    )

    processor.process()
