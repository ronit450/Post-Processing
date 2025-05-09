import numpy as np
from rtree import index
import numpy as np
from decimal import Decimal, getcontext
import os 
import ast
import traceback
import numpy as np
import cv2  
import json 

R = 6378137  

SQUARE_METER = 4046.856


class crop:
    def __init__(self, name, buffer_size):
        self.name = name
        self.buffer_size = buffer_size
    
    def crop_buffer_size(self, name):
        return 
 

class PostProcess:
    '''
    The post process will handle detection conversion and output generation.
    '''
    def main(self, json_path, class_obj_lst, output_path, data, clean_json_path) -> None:
        
        try:
            corners, gsd, width, height, image_name  = self.read_corners_and_gsd_csv(data, json_path)
            Detection_obj = DetectionProcessor(json_path, gsd, class_obj_lst)
            clean_detection = Detection_obj.process_detections(clean_json_path, width, height, image_name, corners, gsd)
            geojson_obj = GeoJSONConverter(output_path, corners, width, height)
            count = geojson_obj.convert_to_geojson(clean_detection)
            return gsd, width, height, count, corners, clean_detection
        except Exception as e:
            traceback.print_exc()
            print(f"Error occured in {json_path}: {str(e)}")
            
        
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
    
      
class DetectionProcessor:
    '''
    Processes and filters detections based on specific criteria.
    '''
    def __init__(self, input_path, gcd, class_obj_lst):
        self.input_path = input_path
        self.gcd = gcd
        self.class_obj_lst = class_obj_lst
        self.detections = self.load_detections()
    def load_detections(self):
        '''
        Loads detections from a JSON file.
        '''
        with open(self.input_path) as f:
            data = json.load(f)
        return data['detections']
    def calculate_center_and_fixed_bbox(self, detections):
        '''
        Calculates center points and fixes bounding box size for each detection.
        '''
        processed_detections = []
        unprocessed_detections = []
        for det in detections:
        
            box = np.array([det['box']['x1'], det['box']['y1'], det['box']['x2'], det['box']['y2']])
            # so now I have the name of that detection and I can easily map it to its size
            box_size = self.class_obj_lst[det['name']]
            half_size = (box_size / 2) / self.gcd
            center = (box[:2] + box[2:]) / 2
            new_box = np.hstack([center - half_size, center + half_size])
            det['box'] = {
                'x1': new_box[0],
                'y1': new_box[1],
                'x2': new_box[2],
                'y2': new_box[3]
            }
            processed_detections.append(det)
        else:
            unprocessed_detections.append(det)
        return processed_detections, unprocessed_detections
    
    def remove_contained_boxes(self, detections):
        """
        Keep only the largest‐class detection for any fully‐contained boxes.
        Uses an R-tree to get O(n log n) performance.
        """
        # 1) sort descending by “real” size so big boxes come first
        dets = sorted(
            detections,
            key=lambda d: self.class_obj_lst.get(d['name'], 0),
            reverse=True
        )

        keep = []
        rtree_idx = index.Index()

        for det in dets:
            # pack into a 4-tuple
            x1, y1 = det['box']['x1'], det['box']['y1']
            x2, y2 = det['box']['x2'], det['box']['y2']
            bbox = (x1, y1, x2, y2)

            # find any previously kept box whose envelope overlaps
            hits = list(rtree_idx.intersection(bbox))

            # check if *any* of those actually fully contains this box
            contained = False
            for idx in hits:
                k = keep[idx]['box']
                if (k['x1'] <= x1 <= x2 <= k['x2'] and
                    k['y1'] <= y1 <= y2 <= k['y2']):
                    contained = True
                    break

            if not contained:
                # no larger box contains it → keep & index
                rtree_idx.insert(len(keep), bbox)
                keep.append(det)

        return keep
    def detect_and_merge(self, detections):
        '''
        Merges overlapping detections based on bounding box intersection using iterative merging with rtree.
        '''
        final_detections = []
        rtree_idx = index.Index()
        for i, det in enumerate(detections):
            bbox = (det['box']['x1'], det['box']['y1'], det['box']['x2'], det['box']['y2'])
            rtree_idx.insert(i, bbox)
        used = [False] * len(detections)
        for i, det1 in enumerate(detections):
            if used[i]:
                continue
            merged_box = det1['box']
            used[i] = True
            merged = True
            while merged:
                merged = False
                overlapping_idxs = list(rtree_idx.intersection((merged_box['x1'], merged_box['y1'], merged_box['x2'], merged_box['y2'])))
                for j in overlapping_idxs:
                    if not used[j] and detections[j]['name'] == det1['name']:
                        # Check if boxes overlap
                        if (merged_box['x1'] < detections[j]['box']['x2'] and
                            merged_box['x2'] > detections[j]['box']['x1'] and
                            merged_box['y1'] < detections[j]['box']['y2'] and
                            merged_box['y2'] > detections[j]['box']['y1']):
                            merged_box['x1'] = min(merged_box['x1'], detections[j]['box']['x1'])
                            merged_box['y1'] = min(merged_box['y1'], detections[j]['box']['y1'])
                            merged_box['x2'] = max(merged_box['x2'], detections[j]['box']['x2'])
                            merged_box['y2'] = max(merged_box['y2'], detections[j]['box']['y2'])
                            used[j] = True
                            merged = True
            final_detections.append({
                'name': det1['name'],
                'box': merged_box,
                'confidence': det1.get('confidence', 1.0), 
            })
        return final_detections
    
    
    def plotter(self, image_path, detections, output_path): 
        img = cv2.imread(image_path)  
        if img is None:
            raise ValueError(f"Failed to load image from {image_path}")

        for det in detections['detections']:
            try:
                x1, y1 = det['box']['x1'], det['box']['y1']
                x2, y2 = det['box']['x2'], det['box']['y2']
                
                x1 = int(round(det['box']['x1']))
                y1 = int(round(det['box']['y1']))
                x2 = int(round(det['box']['x2']))
                y2 = int(round(det['box']['y2']))

                label = det.get('name', 'object')
                confidence = det.get('confidence', 0)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, f"{label} {confidence:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            except KeyError as e:
                print(f"Missing key in detection: {e}")

        success = cv2.imwrite(output_path, img)
        if not success:
            raise IOError(f"Failed to write image to {output_path}")

                

    def calculate_center(self, detections):
        '''
        Computes the center coordinates of each bounding box and structures the output.
        '''
        processed_detections = []
        for det in detections:
            box = det['box']
            x_center = (box['x1'] + box['x2']) / 2
            y_center = (box['y1'] + box['y2']) / 2
            processed_detections.append({
                'name': det['name'],
                'coordinates': [x_center, y_center],
                'box': { 
                    'x1': box['x1'],
                    'y1': box['y1'],
                    'x2': box['x2'],
                    'y2': box['y2']
                },
                'confidence': det.get('confidence', 1.0),
            })
        return processed_detections
    
    def calculate_image_center(self, corners):
        latitudes = [corner[1] for corner in corners]
        longitudes = [corner[0] for corner in corners]

        center_lat = sum(latitudes) / len(latitudes)
        center_lon = sum(longitudes) / len(longitudes)
        
        return center_lat, center_lon
            
    def process_detections(self, clean_json_path, width, height, image_name, corners, gsd):
        '''
        Processes detections and returns cleaned results.
        '''
        processed_detections, unprocessed_detections = self.calculate_center_and_fixed_bbox(self.detections)
        # print(processed_detections)
        combined_detections = processed_detections + unprocessed_detections
        merged_detections = self.detect_and_merge(combined_detections)
        filtered = self.remove_contained_boxes(merged_detections)
        
        center_lat, center_lon = self.calculate_image_center(corners)
        center_detection = self.calculate_center(processed_detections)
        
        final_json = {
            "ImageHeight": height,
            "ImageWidth": width,
            "ImagePath": image_name,
            "Image_center": (center_lat, center_lon), \
            'gsd': gsd,
            "detections": center_detection
        }
        with open(clean_json_path, 'w') as outfile:
            # because in final jsons we only need the pt detections
            json.dump(final_json, outfile, indent=4)
        return {'detections': filtered}
    

class GeoJSONConverter:
    '''
    Converts detection data to GeoJSON with geospatial coordinates.
    '''
    def __init__(self, output_path, corners, image_width, image_height):
        self.output_path = output_path
        self.image_height = image_height
        self.image_width = image_width
        getcontext().prec = 48
        self.gps_corners = {
            "top_left": corners[0],
            "top_right": corners[1],
            "bottom_right": corners[2],
            "bottom_left": corners[3]
        }
    
    def interpolate_to_gps(self, x, y):
        '''
        Interpolates image pixel coordinates to GPS coordinates with high precision.
        Returns Decimal values for maximum precision.
        '''
        norm_x = Decimal(x) / Decimal(self.image_width)
        norm_y = Decimal(y) / Decimal(self.image_height)
        
        lon = (
            Decimal(self.gps_corners["top_left"][0]) * (1 - norm_x) * (1 - norm_y) +
            Decimal(self.gps_corners["top_right"][0]) * norm_x * (1 - norm_y) +
            Decimal(self.gps_corners["bottom_right"][0]) * norm_x * norm_y +
            Decimal(self.gps_corners["bottom_left"][0]) * (1 - norm_x) * norm_y
        )
        
        lat = (
            Decimal(self.gps_corners["top_left"][1]) * (1 - norm_x) * (1 - norm_y) +
            Decimal(self.gps_corners["top_right"][1]) * norm_x * (1 - norm_y) +
            Decimal(self.gps_corners["bottom_right"][1]) * norm_x * norm_y +
            Decimal(self.gps_corners["bottom_left"][1]) * (1 - norm_x) * norm_y
        )
        
        return f"{lat:.42f}", f"{lon:.42f}"
    
    def convert_to_geojson(self, data):
        area_in_sq_m = round(self.image_width * self.image_height, 3)
        count = 0
        features = []
        
        for detection in data['detections']:
            x1, y1 = detection['box']['x1'], detection['box']['y1']
            x2, y2 = detection['box']['x2'], detection['box']['y2']
            center_y = (y1 + y2) / 2
            

            center_x = (x1 + x2) / 2
            lat_str, lon_str = self.interpolate_to_gps(center_x, center_y)
            feature = self._create_point_feature(detection, lon_str, lat_str)
            count += 1
            

            features.append(feature)
        
        per_acre_production = count / (area_in_sq_m / 4046.85642)
        geojson = self._build_geojson(features, area_in_sq_m, per_acre_production)
        
        with open(self.output_path, 'w') as geojson_file:
            geojson_file.write(geojson)
        
        return count 

    def _create_point_feature(self, detection, lon_str, lat_str):
        feature = [
            '    {',
            '      "type": "Feature",',
            '      "geometry": {',
            '        "type": "Point",',
            f'        "coordinates": [{lon_str}, {lat_str}]',
            '      },',
            '      "properties": {',
            f'        "name": "{detection["name"]}",',
            f'        "confidence": {detection["confidence"]},',
            '        "type": "Point"',
            '      }',
            '    }'
        ]
        return '\n'.join(feature)

    def _create_line_feature(self, detection, left_lon, left_lat, right_lon, right_lat):
        feature = [
            '    {',
            '      "type": "Feature",',
            '      "geometry": {',
            '        "type": "LineString",',
            f'        "coordinates": [[{left_lon}, {left_lat}], [{right_lon}, {right_lat}]]',
            '      },',
            '      "properties": {',
            f'        "name": "{detection["name"]}",',
            f'        "confidence": {detection["confidence"]},',
            '        "type": "Line"',
            '      }',
            '    }'
        ]
        return '\n'.join(feature)

    def _build_geojson(self, features, area, per_acre):
        features_str = ',\n'.join(features)
        
        geojson_parts = [
            '{',
            '  "type": "FeatureCollection",',
            '  "properties": {',
            f'    "Area": {area},',
            f'    "Per Acre Production": {per_acre}',
            '  },',
            '  "features": [',
            f'{features_str}',
            '  ]',
            '}'
        ]
        
        return '\n'.join(geojson_parts)

class Analysis:
    def __init__(self, field_json, gsd, image_width, image_height, label, count, corners):
        # self.field_json = field_json
        self.image_width = image_width
        self.image_height = image_height
        self.gsd = gsd
        self.label = label
        self.count = count
        self.corners = corners
        
        with open(field_json, 'r') as data:
            self.field_json = json.load(data)
    
    

        
    
    def for_emergence(self, path):
        latitudes = [corner[1] for corner in self.corners]
        longitudes = [corner[0] for corner in self.corners]
        center_lat = sum(latitudes) / len(latitudes)
        center_lon = sum(longitudes) / len(longitudes)
        image_area = (self.image_width * self.gsd) * (self.image_height * self.gsd)
        path = os.path.splitext(path)[0]
        
        image_info = {
        "image_name": path,
        "image_area": image_area,
        "avg plants per square meter": self.count / image_area, 
        "coordinates": [center_lat, center_lon], 
        "plant_count" : self.count
        }
    
        return image_info
       


    def convert_to_geojson(self, maryam_emergence, output_path):
        geojson = {
            "type": "FeatureCollection",
            "features": []
        }

        for emergence_dict in maryam_emergence:
            feature = {
                "type": "Feature",
                "properties": {
                    "image_name": emergence_dict["image_name"],
                    "image_area": emergence_dict["image_area"],
                     "avg plants per square meter" : emergence_dict["avg plants per square meter"], 
                     "plant count" : emergence_dict['plant_count']
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": emergence_dict["coordinates"][::-1]
                }
            }
            geojson["features"].append(feature)

        with open(output_path, "w") as f:
            json.dump(geojson, f, indent=4)

        return output_path
    
        

        
        

    def one_snap_analysis(self):
        type_label = 'PlantCount'
        label = os.path.splitext(os.path.splitext(os.path.basename(self.label))[0])[0]
        total_crop_area_sq = round((self.image_width * self.gsd) * (self.image_height * self.gsd), 2)
        target_population = round(self.field_json.get('target_stand_per_acre') / SQUARE_METER, 3)
        emerged_population = self.count / total_crop_area_sq  # math.ceil if needed
        emergence_rate = emerged_population / target_population * 100
        yield_loss_plants = target_population - emerged_population
        yield_loss_percentage = yield_loss_plants / target_population * 100
    
        color, plant_count = self.get_status_and_color(emergence_rate)
        analysis_results = {
            "type": type_label,
            "label": label,
            "total_crop_area_sq": round(total_crop_area_sq,2),
            "target_population": round(target_population * total_crop_area_sq,0),
            "emerged_population": self.count,
            "emergence_rate": emergence_rate,
            "yield_loss_plants": round(yield_loss_plants * total_crop_area_sq,0),
            "yield_loss_percentage": yield_loss_percentage,
            "color": color,
            "plant_count": plant_count
        }

        return analysis_results 
            
    
    def get_status_and_color(self,target_achieved):
        if target_achieved > 90:
            return "#006400", "Excellent"  # Dark Green
        elif 80 <= target_achieved <= 90:
            return "#008000", "Good"  # Green
        elif 70 <= target_achieved < 80:
            return "#FFFF00", "Average"  # Yellow
        else:
            return "#FF0000", "Poor"  # Red

     
    
    def generate_field_analysis(self, sum_emerged_pop):
        target_population = self.field_json.get('target_stand_per_acre', 1)
        total_crop_area_acres = self.field_json['polygon']['size']
        emerged_population = (sum(sum_emerged_pop) / len(sum_emerged_pop)) * 4046.856
        emergence_rate = (emerged_population / target_population * 100) if target_population else 0
        yield_loss_plants = target_population - emerged_population
        yield_loss_percentage = (yield_loss_plants / target_population * 100) if target_population else 0
        
        color, plant_count = self.get_status_and_color(emergence_rate)

        field_analysis_data = {
            "label": "summary",
            "type": "PlantCount",
            "company": "",
            "field_id": f"Field {self.field_json.get('id', '')}",
            "boundary_acres": round(total_crop_area_acres,2), #Done 
            "crop_type": self.field_json.get('cropName', ""),
            "farm": self.field_json.get('name', ""),
            "plantation_date": self.field_json.get('seeding_date', ""),
            "flight_scan_date": self.field_json.get('flight_scan_date', ""),
            "total_crop_area_acres": round(total_crop_area_acres,2),
            "target_population_per_acre": round(target_population,0),
            "emerged_population_per_acre": round(emerged_population,0),
            "total_emerged_plants": round(emerged_population * total_crop_area_acres,0) ,
            "total_target_plants": round(target_population * total_crop_area_acres,0),
            "emergence_rate": round(emergence_rate,2),
            "yield_loss_plants": round(yield_loss_plants,0),
            "yield_loss_percentage": round(yield_loss_percentage,2),
            "color": color,
            "plant_count": plant_count
        }

        return field_analysis_data

        
        
        
        
        
        
        
        