import json
import math
from itertools import combinations
from typing import List, Tuple, Set


class ImageDistanceFilter:
    def __init__(self, json_path: str, distance_threshold: float):
        self.json_path = json_path
        self.distance_threshold = distance_threshold
        self.data = self.load_json()
        self.image_centroids = self.extract_centroids()

    def load_json(self) -> List[dict]:
        with open(self.json_path, 'r') as f:
            return json.load(f)

    def extract_centroids(self) -> List[Tuple[str, List[float]]]:
        return [(img['label'], img['centroid']) for img in self.data]

    def haversine(self, lon1, lat1, lon2, lat2) -> float:
        R = 6371000
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        d_phi = math.radians(lat2 - lat1)
        d_lambda = math.radians(lon2 - lon1)
        a = math.sin(d_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2)**2
        return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def remove_close_duplicates(self) -> Set[str]:
        """Removes one image from each pair that is within distance threshold."""
        removed = set()
        for (label1, c1), (label2, c2) in combinations(self.image_centroids, 2):
            if label1 in removed or label2 in removed or label1 == label2:
                continue
            distance = self.haversine(c1[0], c1[1], c2[0], c2[1])
            if distance <= self.distance_threshold:
                removed.add(label2)
        return removed

    def remove_label_duplicates(self) -> List[dict]:
        """Keeps only one entry per label (even if exact duplicates exist)."""
        unique_labels = set()
        deduped = []
        for item in self.data:
            label = item['label']
            if label not in unique_labels:
                unique_labels.add(label)
                deduped.append(item)
        return deduped

    def filter_and_save(self):
        # Step 1: Remove exact duplicates based on label
        deduped_data = self.remove_label_duplicates()

        # Step 2: Reload centroids from deduped data
        self.data = deduped_data
        self.image_centroids = [(img['label'], img['centroid']) for img in deduped_data]

        # Step 3: Remove one from each close pair
        to_remove = self.remove_close_duplicates()
        print(f"Removing {len(to_remove)} images due to proximity")

        final_data = [img for img in deduped_data if img['label'] not in to_remove]

        with open(self.json_path, 'w') as f:
            json.dump(final_data, f, indent=2)

        print(f"✅ Saved filtered JSON. Final image count: {len(final_data)}")


# Example usage
if __name__ == "__main__":
    json_input_path = r"C:\Users\User\Downloads\georefrenced_xmp_alignment_tiles_info.json"
    distance_threshold = 3.0  # meters

    processor = ImageDistanceFilter(json_input_path, distance_threshold)
    processor.filter_and_save()
