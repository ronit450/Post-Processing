import json
import os
import time
import concurrent.futures
import multiprocessing
from utils import *
import traceback
import pandas as pd
# from canopy import * 
import math
import shutil
import random 


class Cleaner:
    def __init__(self, json_folder_path, detect_output_folder, box_size, csv_path, field_json, plot_folder = None) -> None:
        self.json_folder_path = json_folder_path
        self.detect_out =detect_output_folder
        self.box_size = box_size
        self.data = pd.read_csv(csv_path)
        self.post_obj = PostProcess()
        self.max_workers = max(1, multiprocessing.cpu_count() - 4)
        self.field_json = field_json
        self.plot_folder = plot_folder
        self.results = []
        self.emerged_pop_count = []
        self.geojson_output = os.path.join(self.detect_out, 'geojsons')
        os.makedirs(self.detect_out, exist_ok=True)
        self.maryam_emergence = []
        self.clean_detection_path = os.path.join(self.detect_out, 'cleaned_jsons')
        self.plot_output = os.path.join(self.detect_out, 'plot_images')
        os.makedirs(self.plot_output , exist_ok=True)
        os.makedirs(self.clean_detection_path, exist_ok=True)
        os.makedirs(self.geojson_output, exist_ok=True)
        self.count = 0
        
    def process_file(self, file):
        """
        Processes a single file, converting it to a shapefile.
        """
        try:
            start_time = time.time()
            json_path = os.path.join(self.json_folder_path, file)
            detection_save_path = os.path.join(self.clean_detection_path, f"{os.path.splitext(os.path.splitext(file)[0])[0]}.json")
            shp_output_base = os.path.join(self.geojson_output, f"{os.path.splitext(os.path.splitext(file)[0])[0]}.geojson")
            gsd, width, height, count, corners = self.post_obj.main(
                json_path=json_path,
                box_size=self.box_size,
                output_path=shp_output_base, 
                data = self.data,
                clean_json_path = detection_save_path
            ) 
                                 
            end_time = time.time()
            time_taken = end_time - start_time
            print(f"Processed {file} in {time_taken:.2f} seconds")
            self.analysis_obj = Analysis(self.field_json, gsd, width, height, json_path, count, corners)
            analysis_dict = self.analysis_obj.one_snap_analysis()
            emergence_dict = self.analysis_obj.for_emergence(file)
            self.maryam_emergence.append(emergence_dict)
            self.results.append(analysis_dict)
            self.emerged_pop_count.append(analysis_dict['emerged_population']/analysis_dict['total_crop_area_sq'])
        
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            print(traceback.format_exc())
    
    def process(self):
        """
        Processes all files in the specified folder using a ThreadPoolExecutor.
        """
        # files = [f for f in os.listdir(self.json_folder_path) if f.endswith('.out')]
        files = [f for f in os.listdir(self.json_folder_path) if f.endswith('.json')]
        
        for file in files:
            print("Processing File", file)
            self.process_file(file)  # Run sequentially instead of using threading 
        self.analysis_field_dict = self.analysis_obj.generate_field_analysis(self.emerged_pop_count)
        self.geojson_csv_maker()
        self.plot_random_images()
    
    
    def plot_random_images(self, num_images=5):
        all_images = [f for f in os.listdir(self.plot_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.JPG'))]
        selected_images = random.sample(all_images, k=min(num_images, len(all_images)))
        for image_file in selected_images:
            try:
                image_path = os.path.join(self.plot_folder, image_file)
                detection_json_path = os.path.join(self.clean_detection_path, image_file.split('.')[0] + '.json')
                output_path = os.path.join(self.plot_output, f"plot_{image_file}")
                if os.path.exists(detection_json_path):
                    with open(detection_json_path, 'r') as f:
                        json_data = json.load(f)
                        detections = json_data.get("Image_detections", [])
                    DetectionProcessor.plotter(self, image_path, detections, output_path)
            except Exception:
                print(traceback.format_exc())

    


    
    def geojson_csv_maker(self):
        result_folder = os.path.join(self.detect_out, 'Analysis')
        os.makedirs(result_folder, exist_ok=True )
        final_json = []
        final_json.append(self.analysis_field_dict)
        final_json.extend(self.results)
        json_path = os.path.join(result_folder, 'analysis.json')
        csv_path = os.path.join(result_folder, 'final_analysis.csv')
        maryam_emergence_out = os.path.join(result_folder, 'for_maryam_emergence.geojson')
        self.analysis_obj.convert_to_geojson(self.maryam_emergence, maryam_emergence_out)
        with open(json_path, "w", encoding="utf-8") as json_file:
            json.dump(final_json, json_file, indent=4)
        
        
        
        df = pd.DataFrame([{
        "company": self.analysis_field_dict.get("farm", ""),
        "fieldId": self.analysis_field_dict.get("field_id", ""),
        "boundary": f"{self.analysis_field_dict.get('boundary_acres', '')}",  
        "cropType": self.analysis_field_dict.get("crop_type", "").upper(),
        "plantationDate": self.analysis_field_dict.get("plantation_date", ""),
        "flightScan": self.analysis_field_dict.get("flight_scan_date", ""),
        "seededArea": self.analysis_field_dict.get("total_crop_area_acres", ""),
        "targetPopulation": self.analysis_field_dict.get("total_target_plants", ""),
        "emergedPopulation": self.analysis_field_dict.get("total_emerged_plants", ""),
        "emergenceRate": self.analysis_field_dict.get("emergence_rate", ""),
        "yieldLossPlants": self.analysis_field_dict.get("yield_loss_plants", ""),
        "yieldLossPercentage": self.analysis_field_dict.get("yield_loss_percentage", ""),
        "perAcreTarget": self.analysis_field_dict.get("target_population_per_acre", ""),
        "perAcreEmerged": self.analysis_field_dict.get("emerged_population_per_acre", "")
    }])
        
        df.to_csv(csv_path, index=False)
        print(f"JSON and CSV successfully Done")
        
        
        
    


if __name__ == "__main__":
    box_size = 0.06 # this is in meters
    json_folder =r"C:\Users\User\Downloads\corty-test\test-data"
    post_detection_out =  r"C:\Users\User\Downloads\corty-test\test-data_results"
    csv_path = r"C:\Users\User\Downloads\corty-test\image_details.csv"
    field_json = r"C:\Users\User\Downloads\field_season_shot (7).json"
    plot_folder = r"C:\Users\User\Downloads\corty-test\test-data"
    

    
    processor = Cleaner(
        json_folder, 
        post_detection_out, 
        box_size,
        csv_path, 
        field_json, 
        plot_folder
    )
    processor.process()