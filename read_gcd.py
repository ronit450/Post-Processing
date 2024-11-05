
import piexif
import json 

def read_corners_and_gsd_from_exif(image_path):
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



image_file = r"C:\Users\User\Downloads\Aiman-file\copy_DJI_20240622150315_0011.JPG"
print(read_corners_and_gsd_from_exif(image_file))