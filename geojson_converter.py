import json
import geojson
from PIL import Image
import piexif

class GeoJSONConverter:
    def __init__(self, input_path, output_path, image_path):
        self.input_path = input_path
        self.output_path = output_path
        corners, gcd = self.read_corners(image_path)
        self.top_left = (corners[1][1], corners[1][0])  # lat, lon of NW corner
        self.bottom_right = (corners[3][1], corners[3][0])  
        with Image.open(image_path) as img:
            self.image_width, self.image_height = img.size  

    def read_corners(self, image_path):
        try:
            exif_dict = piexif.load(image_path)
            user_comment = exif_dict["Exif"].get(piexif.ExifIFD.UserComment)
            if user_comment and user_comment.startswith(b"XMP\x00"):
                json_data = user_comment[4:].decode('utf-8')
                metadata = json.loads(json_data)
                return metadata.get("corner_coordinates"), metadata.get("gsd")
        except Exception as e:
            print(f"Error reading metadata from {image_path}: {str(e)}")
        return None, None

    def interpolate_to_gps(self, x, y):
        lon = self.top_left[1] + (x / self.image_width) * (self.bottom_right[1] - self.top_left[1])
        lat = self.top_left[0] + (y / self.image_height) * (self.bottom_right[0] - self.top_left[0])
        return lat, lon

    def convert_to_geojson(self):
        with open(self.input_path) as f:
            data = json.load(f)
        features = []
        for detection in data['detections']:
            x1, y1 = detection['box']['x1'], detection['box']['y1']
            x2, y2 = detection['box']['x2'], detection['box']['y2']
            lat1, lon1 = self.interpolate_to_gps(x1, y1)
            lat2, lon2 = self.interpolate_to_gps(x2, y2)
            polygon = geojson.Polygon([[
                [lon1, lat1], [lon1, lat2], [lon2, lat2], [lon2, lat1], [lon1, lat1]
            ]])
            feature = geojson.Feature(
                geometry=polygon,
                properties={
                    "name": detection['name'],
                    "class": detection['class'],
                    "confidence": detection['confidence']
                }
            )
            features.append(feature)
        feature_collection = geojson.FeatureCollection(features)
        with open(self.output_path, 'w') as geojson_file:
            geojson.dump(feature_collection, geojson_file, indent=4)


image_path = r"C:\Users\User\Downloads\Aiman-file\copy_DJI_20240622150315_0011.JPG"
converter = GeoJSONConverter(
    input_path=r"C:\Users\User\Downloads\New folder\New folder\DJI_20240622150315_0011.JPG.json", 
    output_path=r"C:\Users\User\Downloads\Aiman-file\result.geojson",
    image_path= image_path,

)
converter.convert_to_geojson()
