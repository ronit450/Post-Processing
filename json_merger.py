import os
import json

import os
import json

def concatenate_json_files(folder_path, output_file):
    combined_detections = {"detections": []}  # Initialize with an empty detections list

    # Loop through all JSON files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)

            # Load each JSON file and check if it has the expected structure
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                    # Check if the JSON structure is a dictionary and contains 'detections'
                    if isinstance(data, dict) and "detections" in data:
                        combined_detections["detections"].extend(data["detections"])
                    else:
                        print(f"Skipping file {filename}: Not in expected format. Structure: {type(data)} - {data}")
            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    # Save the combined detections to the output file
    with open(output_file, 'w') as outfile:
        json.dump(combined_detections, outfile, indent=4)


# Example usage:
folder_path = r'C:\Users\User\Downloads\New folder\New folder'  # Replace with your folder path
output_file = r'C:\Users\User\Downloads\New folder\New folder\final.json'  # Replace with the desired output file path

concatenate_json_files(folder_path, output_file)
