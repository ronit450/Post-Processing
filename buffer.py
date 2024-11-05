import json
import numpy as np
import os
import time
from utils import *

class Cleaner:
    def __init__(self, folder_path, output_folder, box_size, overlap_threshold) -> None:
        self.folder_path = folder_path
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)
        self.box_size = box_size
        self.overlap_threshold = overlap_threshold

    def process(self):
        for file in os.listdir(self.folder_path):
            if file.endswith('.JPG') or file.endswith('.tif'):
                start_time = time.time()
                image_file = os.path.join(self.folder_path, file)
                json_file = os.path.splitext(file)[0] + '.json'
                print(f"Processing {json_file}")
                json_path = os.path.join(self.folder_path, json_file)
                
                shp_output_base = os.path.join(self.output_folder, os.path.splitext(file)[0])
                
                PostProcess(
                    image_path=image_file, 
                    json_path=json_path, 
                    box_size=self.box_size, 
                    overlap_threshold=self.overlap_threshold, 
                    output_path=shp_output_base  
                )
                
                end_time = time.time()
                time_taken = end_time - start_time
                print(f"Processed {file} in {time_taken:.2f} seconds")

if __name__ == "__main__":
    box_size_cm = 4  # Size of potato in cm
    overlap_threshold = 0.8  # Overlap threshold for merging

    processor = Cleaner(
        r"C:\Users\User\Downloads\Aiman-file",  # Input folder path
        r"C:\Users\User\Downloads\Aiman-file-results",  # Output folder path
        box_size_cm,
        overlap_threshold
    )

    processor.process()
