import json
import os
import time
import concurrent.futures
import multiprocessing
from utils import *
import traceback
import pandas as pd
from canopy import * 
import math


class Cleaner:
    def __init__(self, json_folder_path, detect_output_folder, box_size, csv_path, field_json) -> None:
        self.json_folder_path = json_folder_path
        self.detect_out =detect_output_folder
        self.box_size = box_size
        self.data = pd.read_csv(csv_path)
        self.post_obj = PostProcess()
        self.max_workers = max(1, multiprocessing.cpu_count() - 4)
        self.field_json = field_json
        self.results = []
        self.emerged_pop_count = []
        self.geojson_output = os.path.join(self.detect_out, 'geojsons')
        os.makedirs(self.detect_out, exist_ok=True)
                    
        
    def process_file(self, file):
        """
        Processes a single file, converting it to a shapefile.
        """
        try:
            start_time = time.time()
            json_path = os.path.join(self.json_folder_path, file)
            shp_output_base = os.path.join(self.geojson_output, f"{os.path.splitext(file)[0]}.geojson")
            gsd, width, height, count = self.post_obj.main(
                json_path=json_path,
                box_size=self.box_size,
                output_path=shp_output_base, 
                data = self.data, 
                field_json= self.field_json
            )                   
            end_time = time.time()
            time_taken = end_time - start_time
            print(f"Processed {file} in {time_taken:.2f} seconds")
            self.analysis_obj = Analysis(self.field_json, gsd, width, height, json_path, count)
            analysis_dict = self.analysis_obj.one_snap_analysis()
            self.results.append(analysis_dict)
            self.emerged_pop_count.append(analysis_dict['emerged_population'])
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            print(traceback.format_exc())
    
    def process(self):
        """
        Processes all files in the specified folder using a ThreadPoolExecutor.
        """
        files = [f for f in os.listdir(self.json_folder_path) if f.endswith('.out')]
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.process_file, file) for file in files]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error in thread: {str(e)}")
        
        self.analysis_field_dict = self.analysis_obj.generate_field_analysis(self.emerged_pop_count)
        self.json_csv_maker()
    
    
    def json_csv_maker(self):
        result_folder = os.path.join(self.detect_out, 'Analysis')
        os.makedirs(result_folder, exist_ok=True )
        final_json = []
        final_json.append(self.analysis_field_dict)
        final_json.append(self.results)
        json_path = os.path.join(result_folder, 'analysis.json')
        csv_path = os.path.join(result_folder, 'final_analysis.csv')
        with open(json_path, "w", encoding="utf-8") as json_file:
            json.dump(final_json, json_file, indent=4)
        
        
        df = pd.DataFrame([{
        "company": self.analysis_field_dict.get("company", ""),
        "fieldId": self.analysis_field_dict.get("field_id", ""),
        "boundary": f"{self.analysis_field_dict.get('boundary_acres', '')} Acres",  
        "cropType": self.analysis_field_dict.get("crop_type", ""),
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
    box_size = 0.04 # this is in meters
    json_folder = r"C:\Users\User\Downloads\Sir-Kaamla"
    post_detection_out =  r"C:\Users\User\Downloads\test_json_result-new"
    csv_path = r"C:\Users\User\Downloads\image_details (1).csv"
    field_json = r"C:\Users\User\Downloads\field_season_shot.json"
    

    
    processor = Cleaner(
        json_folder, 
        post_detection_out, 
        box_size,
        csv_path, 
        field_json
    )
    processor.process()