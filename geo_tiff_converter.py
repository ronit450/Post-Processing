from osgeo import gdal, osr
import os, json, piexif

def georeference_image_with_gcp(images_path, output_folder):
    """
    Georeference JPG images using GDAL with GCPs extracted from EXIF data.

    Parameters:
    images_path (str): Path to the folder containing JPG images.
    output_folder (str): Path to the folder where output GeoTIFF files will be saved.

    Returns:
    list: Paths to the output georeferenced GeoTIFF files.
    """
    # List all JPG files in the directory
    files = [os.path.join(images_path, f) for f in os.listdir(images_path) if f.endswith((".JPG", ".JPEG,", "jpg", "jpeg"))]

    output_tiff_list = []

    for file in files:
        ds = gdal.Open(file)
        if ds is None:
            raise ValueError(f"Unable to open image file: {file}")

        imgW, imgH = ds.RasterXSize, ds.RasterYSize

        # Set the coordinate system (WGS84)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)

        # Extract GCP coordinates from EXIF data
        exif_dict = piexif.load(file)
        user_comment = exif_dict["Exif"].get(piexif.ExifIFD.UserComment)

        if not (user_comment and user_comment.startswith(b"XMP\x00")):
            raise ValueError(f"No valid EXIF coordinates found in {file}")

        json_data = user_comment[4:].decode('utf-8')
        coordinates = json.loads(json_data)

        # Create GCP list
        gcp_list = [
            gdal.GCP(coordinates[0][0], coordinates[0][1], 0, 0, 0),
            gdal.GCP(coordinates[1][0], coordinates[1][1], 0, imgW - 1, 0),
            gdal.GCP(coordinates[2][0], coordinates[2][1], 0, imgW - 1, imgH - 1),
            gdal.GCP(coordinates[3][0], coordinates[3][1], 0, 0, imgH - 1)
        ]

        # Calculate output extent
        min_x = min(coord[0] for coord in coordinates)
        max_x = max(coord[0] for coord in coordinates)
        min_y = min(coord[1] for coord in coordinates)
        max_y = max(coord[1] for coord in coordinates)

        # Define paths
        temp_path = os.path.splitext(file)[0] + '_temp.tif'
        output_path = os.path.join(output_folder, os.path.basename(file)[:-4] + ".tif")

        # Remove existing files if necessary
        for path in [temp_path, output_path]:
            if os.path.exists(path):
                os.remove(path)

        # Step 1: Apply GCPs using gdal.Translate
        gdal.Translate(
            temp_path, ds, format='GTiff', GCPs=gcp_list, outputSRS=srs.ExportToWkt()
        )

        # Step 2: Warp the image with transparency
        gdal.Warp(
            output_path, temp_path, format='GTiff',
            srcSRS=srs.ExportToWkt(), dstSRS=srs.ExportToWkt(),
            outputBounds=[min_x, min_y, max_x, max_y],
            xRes=(max_x - min_x) / imgW, yRes=(max_y - min_y) / imgH,
            tps=True,  # Use Thin Plate Spline transformation
            resampleAlg=gdal.GRA_Bilinear,
            outputType=gdal.GDT_Byte,
            dstAlpha=True,  # Add alpha channel
            dstNodata=0  # Set nodata value to 0 (transparent)
        )

        # Add alpha channel handling
        out_ds = gdal.Open(output_path, gdal.GA_Update)
        alpha_band = out_ds.GetRasterBand(4)
        alpha_data = alpha_band.ReadAsArray()
        alpha_data[alpha_data == 0] = 0  # Transparent
        alpha_data[alpha_data > 0] = 255  # Opaque
        alpha_band.WriteArray(alpha_data)

        # Close datasets
        out_ds = None
        ds = None

        # Remove temporary file
        os.remove(temp_path)

        # Collect output paths
        output_tiff_list.append(output_path)

    return output_tiff_list

images = r"C:\Users\User\Downloads\Aiman-file"
out_img = r"C:\Users\User\Downloads\Aiman-file"

georeference_image_with_gcp(images, out_img)