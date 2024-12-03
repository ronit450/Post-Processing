import json
import os
import time
import concurrent.futures
import multiprocessing
from utils import *
import traceback

class Cleaner:
    def __init__(self, img_folder_path, json_folder_path, output_folder, box_size) -> None:
        self.img_folder_path = img_folder_path
        self.json_folder_path = json_folder_path
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)
        self.box_size = box_size
        self.max_workers = max(1, multiprocessing.cpu_count() - 4)
    def process_file(self, file):
        """
        Processes a single file, converting it to a shapefile.
        """
        try:
            if file.endswith('.JPG') or file.endswith('.tif'):
                start_time = time.time()
                image_file = os.path.join(self.img_folder_path, file)
                json_path = os.path.join(self.json_folder_path, file.replace('.JPG', '.json'))
                shp_output_base = os.path.join(self.output_folder, f"{os.path.splitext(file)[0]}.geojson")
                PostProcess(
                    image_path=image_file,
                    json_path=json_path,
                    box_size=self.box_size,
                    output_path=shp_output_base
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
        files = [f for f in os.listdir(self.img_folder_path) if f.endswith('.JPG') or f.endswith('.tif')]
        for file in files:
            print(file)
            self.process_file(file)


if __name__ == "__main__":
    box_size_cm = 4
    processor = Cleaner(

        r"C:\Users\User\Downloads\pp\imahes", 
        r"C:\Users\User\Downloads\pp\json", 
        r"C:\Users\User\Downloads\pp\out",  # Output folder path
        box_size_cm,
    )
    processor.process()