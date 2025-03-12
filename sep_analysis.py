from utils import * 
import pandas as pd

class sepAnalysis:
    def __init__(self, geojson_folder, field_json, image_details):
        self.geojson_folder = geojson_folder
        self.field_json = field_json
        self.image_details = pd.read_csv(image_details)
        
    def process_file(self):
        
        corners, gsd, width, height, image_name  = self.read_corners_and_gsd_csv(data, json_path)
        
    def process(self):

        files = [f for f in os.listdir(self.json_folder_path) if f.endswith('.geojson')]
        
        for file in files:
            print("Processing File", file)
            self.process_file(file)  # Run sequentially instead of using threading 
        self.analysis_field_dict = self.analysis_obj.generate_field_analysis(self.emerged_pop_count)
       
        self.geojson_csv_maker()