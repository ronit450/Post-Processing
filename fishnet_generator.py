import os
from maryam_utils import *

class SpatialAnalysisProcessor:
    def __init__(self, points_geojson, boundary_geojson, output_folder, resolution, fishnet_cell_size, target_pop):
        self.geo_processor = GeoSpatialProcessor(points_geojson, boundary_geojson, output_folder, resolution, fishnet_cell_size)
        dissolved_geojson = os.path.join(self.geo_processor.OUTPUT_FOLDER, "dissolved.geojson")
        self.zonal_processor = ZonalStatisticsProcessor(None, None, target_pop, dissolved_geojson)
        
    def run_analysis(self):
        # Step 1: Run GeoSpatialProcessor to create required files
        print("Running GeoSpatialProcessor...")
        self.geo_processor.generate_idw_raster()
        
        # Set output paths from GeoSpatialProcessor
        idw_output_clipped = os.path.join(self.geo_processor.OUTPUT_FOLDER, "idw_output_clipped.tif")
        fishnet_clipped = os.path.join(self.geo_processor.OUTPUT_FOLDER, "summary.geojson")
      
        
        # Step 2: Update ZonalStatisticsProcessor with the new files
        self.zonal_processor.raster_path = idw_output_clipped
        self.zonal_processor.clipped_fishnet_path = fishnet_clipped
        
        # Step 3: Run ZonalStatisticsProcessor to extract zonal mean and dissolve by class
        print("Running ZonalStatisticsProcessor...")
        self.zonal_processor.extract_zonal_mean()
        self.zonal_processor.dissolve_by_class()
        
        print("Analysis complete. Final output saved at:", self.zonal_processor.output_geojson)


if __name__ == "__main__":
    processor = SpatialAnalysisProcessor(
        points_geojson=r"C:\Users\User\Downloads\for_maryam_emergence.geojson",
        boundary_geojson=r"C:\Users\User\Downloads\0262-Siro-Mack_1906.geojson", 
        output_folder=r"C:\Users\User\Downloads\maryam-test", 
        resolution=0.00000898,
        fishnet_cell_size=0.000089,
        target_pop=4.88, #Per m2
    )
    processor.run_analysis()
