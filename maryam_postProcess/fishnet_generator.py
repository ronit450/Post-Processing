import os
from maryam_utils import *

class SpatialAnalysisProcessor:
    def __init__(self, points_geojson, boundary_geojson, output_folder, resolution, fishnet_cell_size, target_pop):
        self.boundary_geojson = 'temp_boundary.geojson'
        wkt = self.read_json(boundary_geojson)
        self.wkt_to_geojson(wkt, self.boundary_geojson)
        
        self.geo_processor = GeoSpatialProcessor(points_geojson, self.boundary_geojson, output_folder, resolution, fishnet_cell_size)
        dissolved_geojson = os.path.join(self.geo_processor.OUTPUT_FOLDER, "dissolved.geojson")
        self.zonal_processor = ZonalStatisticsProcessor(None, None, target_pop, dissolved_geojson)
    
    def read_json(self, file_path):
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                return data.get('polygon', {}).get('geometry_wkt')
        except FileNotFoundError:
            print("Error: File not found.")
        except json.JSONDecodeError:
            print("Error: Invalid JSON format.")

    def wkt_to_geojson(self, wkt_string, output_geojson):
        is_3d = "POLYGON Z" in wkt_string
        
        # Parse WKT using OGR
        geometry = ogr.CreateGeometryFromWkt(wkt_string)

        if geometry is None or geometry.GetGeometryName() not in ["POLYGON", "POLYGON25D"]:
            print("❌ Invalid WKT polygon.")
            return

        # Ensure the polygon is closed
        ring = geometry.GetGeometryRef(0)  # Get outer ring
        first_point = ring.GetPoint(0)
        last_point = ring.GetPoint(ring.GetPointCount() - 1)

        if first_point[:2] != last_point[:2]:  # Check only X, Y for closure
            print("⚠️ AOI WKT is not closed. Closing it automatically.")
            ring.AddPoint(*first_point)  # Re-add first point

        # Prepare points to only include X and Y (omit Z if present)
        points = []
        for i in range(ring.GetPointCount()):
            point = ring.GetPoint(i)
            points.append([point[0], point[1]])  # Only append X and Y

        # Create GeoJSON geometry
        geojson_geometry = {
            "type": "Polygon",
            "coordinates": [points]  # Ensure it’s in a list of list format
        }

        # Create GeoJSON feature
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"id": 1},
                    "geometry": geojson_geometry
                }
            ]
        }
        
        # Write to file
        with open(output_geojson, 'w') as f:
            json.dump(geojson, f, indent=4)
        
    def run_analysis(self):
        # Step 1: Run GeoSpatialProcessor to create required files
        print("Running GeoSpatialProcessor...")
        self.geo_processor.generate_idw_raster()
        
        # Set output paths from GeoSpatialProcessor
        idw_output_clipped = os.path.join(self.geo_processor.OUTPUT_FOLDER, "idw_output_clipped.tif")
        fishnet_clipped = os.path.join(self.geo_processor.OUTPUT_FOLDER, "fishnet_clipped.geojson")
      
        
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
        boundary_geojson=r"C:\Users\User\Downloads\field_season_shot (7).json", 
        output_folder=r"C:\Users\User\Downloads\for_maryam_emergence_result",
        resolution=0.00000898,
        fishnet_cell_size=0.000089,
        target_pop=4.88,
    )
    processor.run_analysis()
