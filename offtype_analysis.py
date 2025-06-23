import json
import os
import time
import concurrent.futures
import multiprocessing
from offtype_utils import *
import traceback
import pandas as pd
# from canopy import * 
import math
import shutil
import random 


class Cleaner:
    def __init__(self, json_folder_path, detect_output_folder, field_json, csv_folder, buffer_list) -> None:
        self.json_folder_path = json_folder_path
        self.detect_out =detect_output_folder
        self.field_json = field_json
        self.geojson_output = os.path.join(self.detect_out, 'geojsons')
        self.clean_detection_path = os.path.join(self.detect_out, 'cleaned_jsons')
        self.plot_output = os.path.join(self.detect_out, 'plot_images')
        os.makedirs(self.plot_output, exist_ok=True)
        os.makedirs(self.clean_detection_path, exist_ok=True)
        
        os.makedirs(self.detect_out, exist_ok=True)
        os.makedirs(self.geojson_output, exist_ok=True)
        self.count = 0
        self.data = self.csv_merger(csv_folder)
        self.results = []
        self.total_image_area= 0
        self.image_count = 0
        self.buffer_list = buffer_list

    
    
    def csv_merger(self, csv_folder):
        csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]
        df = pd.concat([pd.read_csv(os.path.join(csv_folder, file)) for file in csv_files], ignore_index=True)
        return df 
    
    def read_corners_and_gsd_csv(self, data, json_path):
        try:
            image_name = os.path.basename(json_path)
            image_name = image_name.replace('.json', '.JPG')
            row = data[data['image_name'] == image_name]
            if not row.empty:
                coordinates = (row.iloc[0]['corners'])
                coordinates = ast.literal_eval(coordinates)
                gsd = row.iloc[0]['gsd']
                width = int(row.iloc[0]['image_width'])
                height = int(row.iloc[0]['image_height'])
                # print(f"this Image: {image_name}, GSD: {gsd}, Width: {width}, Height: {height}, Coordinates: {coordinates}")
                return coordinates, gsd, width, height , image_name
        except Exception as e:
            print(f"Error reading metadata from {json_path}: {str(e)}")
        return None, None, None, None, None
    
    def process_file(self, json_path, image_name):
        """
        Processes a single file, converting it to a shapefile.
        """
        try:
            shp_output_base = os.path.join(self.geojson_output, f"{os.path.splitext(image_name)[0]}.geojson")
            clean_json_path = os.path.join(self.clean_detection_path, f"{os.path.splitext(image_name)[0]}.json")
            corners, gsd, width, height, _ = self.read_corners_and_gsd_csv(self.data, json_path)
            det_processor_obj = DetectionProcessor(json_path, gsd, self.buffer_list)
            clean_detection = det_processor_obj.process_detection(clean_json_path, width, height, image_name, corners, gsd)
            geojson_obj = GeoJSONConverter(shp_output_base, corners, width, height)
            count = geojson_obj.convert_to_geojson(clean_detection)
            self.count += count
            self.analysis_obj = Analysis(self.field_json, gsd, width, height, json_path, count, corners)
            analysis_dict, image_area = self.analysis_obj.one_snap_analysis()
            self.total_image_area += image_area
            self.image_count += 1
            self.results.append(analysis_dict)

        except Exception as e:
            print(f"Error processing {json_path}: {str(e)}")
            print(traceback.format_exc())


    def process(self):
        """
        Processes based on image names from CSV and their corresponding JSONs.
        """
        for _, row in self.data.iterrows():
            image_name = row['image_name']
            json_name = image_name.replace('.JPG', '.json')
            json_path = os.path.join(self.json_folder_path, json_name)

            print("Processing Image", image_name)

            if os.path.exists(json_path):
                self.process_file(json_path, image_name)
            else:
                try:
                    # Handle missing JSON by adding a zero-analysis entry
                    corners = ast.literal_eval(row['corners'])
                    gsd = row['gsd']
                    width = int(row['image_width'])
                    height = int(row['image_height'])
                    self.analysis_obj = Analysis(self.field_json, gsd, width, height, json_path, 0, corners)
                    analysis_dict, image_area = self.analysis_obj.one_snap_analysis()
                    self.total_image_area += image_area
                    self.results.append(analysis_dict)
                except Exception as e:
                    print(f"Error handling missing JSON for {image_name}: {str(e)}")

        self.analysis_field_dict = self.analysis_obj.generate_field_analysis(self.count, self.total_image_area, self.image_count)
        
        result_folder = os.path.join(self.detect_out, 'Analysis')
        os.makedirs(result_folder, exist_ok=True)
        json_path = os.path.join(result_folder, 'offtype_report.json')
        final_json = [self.analysis_field_dict] + self.results

        with open(json_path, "w", encoding="utf-8") as json_file:
            json.dump(final_json, json_file, indent=4)


if __name__ == "__main__":
   
    json_folder = r"C:\Users\User\Downloads\Compressed\2491-20250526T141426Z-1-001\2491\filtered_jsons\filtered_jsons"
    post_detection_out = r"C:\Users\User\Downloads\Compressed\2491-20250526T141426Z-1-001\2491\filtered_jsons\filtered_jsons_result"
    csv_folder = r"C:\Users\User\Downloads\Compressed\images_csv\images_csv"
    field_json = r"C:\Users\User\Downloads\field_season_shot (3).json"
    
    class_obj_lst = {
        'cn_coty' : 0.01,
        'cn_4L' : 0.02 
    
    }
    
    
    processor = Cleaner(
        json_folder, 
        post_detection_out, 
        field_json,
        csv_folder, 
        buffer_list=class_obj_lst
    )
    processor.process()