import os
import shutil
from buffer import * 
from sahi_rk import * 
import pandas as pd
import traceback


def gsd_correcter(cord_csv, image_csv, new_detail_csv):
    
    image_csv = pd.read_csv(image_csv)
    cord_csv = pd.read_csv(cord_csv)
    cord_csv.columns = cord_csv.columns.str.strip()
    image_csv.columns = image_csv.columns.str.strip()

    # Rename column to match for merging
    cord_csv.rename(columns={'image': 'image_name'}, inplace=True)

    # Merge and update GSD values
    image_details_updated = image_csv.drop(columns=['gsd']).merge(
        cord_csv[['image_name', 'gsd']], on='image_name', how='left'
    )
    image_details_updated.to_csv(new_detail_csv, index=False)
    # print(f'updated {cord_csv}')

def process_folders(main_folder, model_path, phsz_width, phsz_height, imgsz, task, output_folder):
    failed_farms = []  

    for folder in os.listdir(main_folder):
        folder_path = os.path.join(main_folder, folder)
        if os.path.isdir(folder_path):  # Ensure it's a directory
            for subfolder in os.listdir(folder_path):
                subfolder_path = os.path.join(folder_path, subfolder)

                
                if os.path.isdir(subfolder_path):
                    try:
                        output_dir_geojson = os.path.join(output_folder, "New-All-Farms-Analysis", subfolder,  "cleaned_geojsons_rk")
                        os.makedirs(output_dir_geojson, exist_ok=True)
                        jpg_folder_path = os.path.join(subfolder_path, "jpg")
                        new_csv_name = f"{subfolder}_image_details.csv"
                        cord_csv = os.path.join(subfolder_path, f"{subfolder}_image_coordinates_new.csv")
                        csv_path = os.path.join(subfolder_path, new_csv_name)
                        
                        new_detail_rk =  os.path.join(subfolder_path, f"{subfolder}_image_details_rk.csv")
                        if str(subfolder) not in ['1881']:
                            continue
                        gsd_correcter(cord_csv, csv_path, new_detail_rk)
                        
                        field_json = os.path.join(subfolder_path, "field_season_shot.json")

                        if os.path.exists(jpg_folder_path) and os.path.isdir(jpg_folder_path):
                            output_dir_json = os.path.join(subfolder_path, "detected_jsons")
                            print(f"Starting for {subfolder}")
                            try:
                                processor1 = SahiDetect(
                                    image_folder=jpg_folder_path,
                                    model_path=model_path,
                                    phsz_width=phsz_width,
                                    phsz_height=phsz_height,
                                    imgsz=imgsz,
                                    output_dir=output_dir_json,
                                    task=task,
                                    csv_path=new_detail_rk
                                )
                                processor1.process_images()
                                print(f"Detection for {subfolder} complete")
                            except Exception as e:
                                print(f"Detection failed for {subfolder}: {e}")
                                failed_farms.append(subfolder)
                                continue  # Skip further processing for this farm

                            try:
                                processor2 = Cleaner(
                                    json_folder_path=output_dir_json, 
                                    detect_output_folder=output_dir_geojson, 
                                    box_size=0.08, 
                                    csv_path=new_detail_rk, 
                                    field_json=field_json, 
                                )
                                processor2.process()
                                print(f"Cleaning for {subfolder} complete")
                            except Exception as e:
                                print(f"Cleaning failed for {subfolder}: {e}")
                                failed_farms.append(subfolder)
                                continue  # Skip further processing for this farm

                            print(f"Processing completed successfully for: {subfolder}")

                    except Exception as e:
                        print(f"Error processing {subfolder}: {e}")
                        failed_farms.append(subfolder)
                        traceback.print_exc()

    # Print the list of failed farms at the end
    if failed_farms:
        print("\nFarms where detection/cleaning failed:")
        for farm in failed_farms:
            print(f"- {farm}")

# Example Usage
main_folder = r"K:\FLL"  # Change this to your actual path
model_path = r"D:\Momna_potato\black-gold\dataset\model_1600\pt_20250313_080217.pt"  # Update accordingly
phsz_width = 4
phsz_height = 3
imgsz = 600
task = 0
output_dir = r"K:" 

process_folders(main_folder, model_path, phsz_width, phsz_height, imgsz, task, output_dir)
