import os
import json

def convert_annotation_file(path, default_class=0, default_confidence=1.0):
    data = json.load(open(path))
    detections = []
    for shape in data.get('shapes', []):
        pts = shape.get('points', [])
        if len(pts) != 2:
            continue
        (x1, y1), (x2, y2) = pts
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        detections.append({
            "name": shape.get("label", ""),
            "class": default_class,
            "confidence": default_confidence,
            "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        })
    return {"detections": detections}

input_dir = r"C:\Users\User\Downloads\2images_FK\2images_FK"

for fname in os.listdir(input_dir):
    if not fname.lower().endswith('.json'):
        continue
    src = os.path.join(input_dir, fname)
    result = convert_annotation_file(src)
    dst = os.path.join(input_dir, os.path.splitext(fname)[0] + ".json")
    json.dump(result, open(dst, 'w'), indent=2)
