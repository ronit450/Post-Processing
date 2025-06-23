import os
import json
import geopandas as gpd
# ==== USER INPUT SECTION ====
geojson_folder = r"C:\Users\Hassan\Downloads\ahmed\2439\offtype"
field_shot_path = r"C:\Users\Hassan\Downloads\ahmed\2439\field_season_shot (19).json"
plant_count_report_path = r"C:\Users\Hassan\Downloads\plant_count_report (6).json"
company = "Hytech"
farm_name = "Slingerland SE-28-9-19"
image_link = ""
# ============================
# Load field shot data
with open(field_shot_path) as f:
    field_data = json.load(f)
# Collect all geojson file labels
geojson_labels = {
    os.path.splitext(f)[0] for f in os.listdir(geojson_folder)
    if f.endswith(".geojson")
}
# Load and standardize plant count report
try:
    with open(plant_count_report_path) as f:
        raw_plant_data = json.load(f)
    plant_data = []
    for entry in raw_plant_data:
        label = entry.get("label", "")
        cleaned_label = os.path.splitext(label)[0]
        if cleaned_label in geojson_labels:
            entry["label"] = cleaned_label
        plant_data.append(entry)
except:
    plant_data = []
# Class thresholds (per sqm)
thresholds = {
    "Hytech": [(0.00247, 0), (0.00123, 1), (0.00025, 2), (0, 3)],
    "Corteva": [(0.01236, 0), (0.00519, 1), (0.00025, 2), (0, 3)],
    "Bayer": [(0.01236, 0), (0.00618, 1), (0.00123, 2), (0, 3)],
    "Nutrien": [(0.00247, 0), (0.00123, 1), (0.00025, 2), (0, 3)],
}
def assign_class(count, company):
    for limit, cls in thresholds.get(company, []):
        if count > limit:
            return cls
    return 3
# Count only Point geometries
def count_valid_points(gdf):
    return gdf[gdf.geometry.type == 'Point'].shape[0]
results = []
summary_count = 0
# Filter usable labels
plant_labels = [entry for entry in plant_data if entry.get("type") == "PlantCount" and entry.get("label") != "summary"]
if not plant_labels:
    print(":warning: No valid PlantCount labels found. Using fallback based on .geojson content...")
    for filename in os.listdir(geojson_folder):
        if filename.endswith(".geojson"):
            label = os.path.splitext(filename)[0]
            path = os.path.join(geojson_folder, filename)
            try:
                gdf = gpd.read_file(path)
                count = count_valid_points(gdf)
                summary_count += count
                results.append({
                    "label": label,
                    "type": "OffType",
                    "total_crop_area_sq": 0,
                    "crop_type": field_data.get("cropName", "").capitalize(),
                    "class": assign_class(count, company),
                    "offtype": count
                })
                print(f":white_tick: {label}: {count} points")
            except Exception as e:
                print(f":x: Failed reading {label}: {e}")
else:
    geojson_counts = {}
    for filename in os.listdir(geojson_folder):
        if filename.endswith(".geojson"):
            label = os.path.splitext(filename)[0]
            path = os.path.join(geojson_folder, filename)
            try:
                gdf = gpd.read_file(path)
                geojson_counts[label] = count_valid_points(gdf)
            except Exception as e:
                print(f":x: Failed reading {label}: {e}")
    for entry in plant_labels:
        label = entry["label"]
        area = entry.get("total_crop_area_sq", 0)
        count = geojson_counts.get(label, 0)
        summary_count += count
        results.append({
            "label": label,
            "type": "OffType",
            "total_crop_area_sq": area,
            "crop_type": field_data.get("cropName", "").capitalize(),
            "class": assign_class(count, company),
            "offtype": count
        })
        print(f":white_tick: {label}: {count} points")
# Compose summary
summary = {
    "label": "summary",
    "type": "OffType",
    "company": f"{company} Production",
    "field_id": field_data.get("name", ""),
    "seeded_area": field_data.get("polygon", {}).get("size", 0),
    "crop_type": field_data.get("cropName", "").capitalize(),
    "farm": farm_name,
    "seeded_date": field_data.get("seeding_date", ""),
    "flight_scan_date": field_data.get("flight_scan_date", ""),
    "off_type_Count": summary_count,
    "image": image_link
}
# Save output
output_file = os.path.join(geojson_folder, "offtype_report.json")
with open(output_file, "w") as f:
    json.dump([summary] + results, f, indent=4)
print(f":white_tick: Final report saved to: {output_file}")