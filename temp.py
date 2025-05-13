#!/usr/bin/env python3
import os
import json
from pathlib import Path

def convert_json_format(input_file, output_file):
    """
    Convert detection JSON format to AnyLabeling format
    """
    # Read input JSON
    with open(input_file, 'r') as f:
        input_data = json.load(f)
    
    # Create output format structure
    output_data = {
        "version": "0.3.3",
        "flags": {},
        "shapes": [],
        "imagePath": input_data["ImagePath"],
        "imageData": None,
        "imageHeight": input_data["ImageHeight"],
        "imageWidth": input_data["ImageWidth"]
    }
    
    # Convert detections to shapes
    for detection in input_data["Image_detections"]:
        # Extract box coordinates
        x1 = detection["box"]["x1"]
        y1 = detection["box"]["y1"]
        x2 = detection["box"]["x2"]
        y2 = detection["box"]["y2"]
        
        # Create shape in AnyLabeling format
        shape = {
            "label": detection["name"],
            "text": "",
            "points": [
                [x1, y1],
                [x2, y2]
            ],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {}
        }
        
        output_data["shapes"].append(shape)
    
    # Write output JSON
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)

def main():
    # Define input and output directories
    input_dir = r"C:\Users\User\Downloads\new"
    output_dir = r"C:\Users\User\Downloads\new_result"
    
    input_dir = Path(input_dir)
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Process all JSON files in the input directory
    json_files = list(input_dir.glob("*.json"))
    
    for input_file in json_files:
        output_file = output_dir / input_file.name
        try:
            convert_json_format(input_file, output_file)
            print(f"Converted {input_file.name} successfully")
        except Exception as e:
            print(f"Error processing {input_file.name}: {str(e)}")
    
    print(f"Conversion complete. Processed {len(json_files)} files.")

if __name__ == "__main__":
    main()






