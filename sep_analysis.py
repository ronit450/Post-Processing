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

class SepAnaly:
    def __init__(self, geojson_folder, image_details, detect_output_folder, field_json):
        self.geojson_folder = geojson_folder
        self.detect_out = detect_output_folder
        os.makedirs(self.detect_out, exist_ok=True)
        self.image_details = pd.read_csv(image_details)
        self.field_json = field_json
        self.results = []
        self.emerged_pop_count = []
        self.maryam_emergence = []
        
    
    def read_corners_and_gsd_csv(self, data, json_path):
        try:
            image_name = os.path.basename(json_path)
            image_name = image_name.replace('.geojson', '.JPG')
            row = data[data['image_name'] == image_name]
            if not row.empty:
                coordinates = (row.iloc[0]['corners'])
                coordinates = ast.literal_eval(coordinates)
                gsd = row.iloc[0]['gsd']
                width = int(row.iloc[0]['image_width'])
                height = int(row.iloc[0]['image_height'])
                return coordinates, gsd, width, height , image_name
        except Exception as e:
            print(f"Error reading metadata from {json_path}: {str(e)}")
        return None, None, None, None, None
    
    def countObj(self, geojson_path):
        """
        Counts the number of objects in a GeoJSON file.
        """
        try:
            with open(geojson_path, 'r') as f:
                data = json.load(f)
                num_objects = len(data['features'])
                return num_objects
        except Exception as e:
            print(f"Error counting objects in {geojson_path}: {str(e)}")
            return 0
    
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

    
    def process_file(self, file):
        """
        Processes a single file, converting it to a shapefile.
        """
        try:
            start_time = time.time()
            json_path = os.path.join(self.geojson_folder, file)
            # Read corners and gsd from the CSV file
            corners, gsd, width, height, image_name = self.read_corners_and_gsd_csv(self.image_details, json_path)
            count = self.countObj(json_path)
            
            # Making Analysis Object
            self.analysis_obj = Analysis(self.field_json, gsd, width, height, json_path, count,  corners)
            analysis_dict = self.analysis_obj.one_snap_analysis()
            emergence_dict = self.analysis_obj.for_emergence(file)
            self.maryam_emergence.append(emergence_dict)
            self.results.append(analysis_dict)
            self.emerged_pop_count.append(analysis_dict['emerged_population']/analysis_dict['total_crop_area_sq'])
            
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
        
        files = [f for f in os.listdir(self.geojson_folder) if f.endswith('.geojson')]
        for file in files:
            print("Processing File", file)
            self.process_file(file)

    

        self.analysis_field_dict = self.analysis_obj.generate_field_analysis(self.emerged_pop_count)
        self.geojson_csv_maker()


if __name__ == "__main__":
    # Example usage
    geojson_folder = r"C:\Users\User\Downloads\saad_geojson\Geojson"
    image_details = r"C:\Users\User\Downloads\image_details_chunk1 (1).csv"
    field_json = r"C:\Users\User\Downloads\field_season_shot (18).json"
    detect_output_folder = r"C:\Users\User\Downloads\saad_geojson\Geojson_result"

    sep_analy = SepAnaly(geojson_folder, image_details, detect_output_folder, field_json)
    sep_analy.process()