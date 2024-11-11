import json
import os
import time
import concurrent.futures
import multiprocessing
from utils import *

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
                json_file = os.path.splitext(file)[0] + '.json'
                json_path = os.path.join(self.json_folder_path, json_file)
                shp_output_base = os.path.join(self.output_folder, os.path.splitext(file)[0])
                print(f"Processing {json_file}")
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

    def process(self):
        """
        Processes all files in the specified folder using a ThreadPoolExecutor.
        """
        files = [f for f in os.listdir(self.img_folder_path) if f.endswith('.JPG') or f.endswith('.tif')]
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.process_file, file) for file in files]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error in thread: {str(e)}")

if __name__ == "__main__":
    box_size_cm = 5 

    processor = Cleaner(
        r"E:\blackgold\nutfarm2\DCIM\output_test",  # Input folder path
        r"D:\Aiman\25-10-2024_complete_pipeline_test\yolo_soybean\code\jsons",  # JSON folder path
        r"E:\blackgold\nutfarm2\DCIM\output_geojsons",  # Output folder path
        box_size_cm,
    )

    processor.process()