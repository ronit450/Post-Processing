import os
import json
import numpy as np
from osgeo import gdal, ogr, osr
from scipy.spatial import cKDTree

class GeoSpatialProcessor:
    def __init__(self, points_geojson, boundary_geojson, output_folder, resolution, fishnet_cell_size):
        self.POINTS_GEOJSON = points_geojson
        self.BOUNDARY_GEOJSON = boundary_geojson
        self.OUTPUT_FOLDER = output_folder
        self.RESOLUTION = resolution
        self.FISHNET_CELL_SIZE = fishnet_cell_size
        
        # Automatically create output directory if it doesn't exist
        if not os.path.exists(self.OUTPUT_FOLDER):
            os.makedirs(self.OUTPUT_FOLDER)
            print(f"Created output directory: {self.OUTPUT_FOLDER}")

    def get_crs_from_geojson(self, geojson_file):
        with open(geojson_file, 'r') as f:
            data = json.load(f)

        crs_str = data.get('crs', {}).get('properties', {}).get('name', None)
        srs = osr.SpatialReference()

        if crs_str:
            if "EPSG::" in crs_str:
                epsg_code = int(crs_str.split("EPSG::")[-1])
                srs.ImportFromEPSG(epsg_code)
            else:
                if srs.SetFromUserInput(crs_str) != 0:
                    print(f"Warning: Unable to interpret CRS '{crs_str}'. Defaulting to EPSG:4326.")
                    srs.ImportFromEPSG(4326)
            return srs

        ds = ogr.Open(geojson_file)
        if ds is not None:
            layer = ds.GetLayer()
            srs = layer.GetSpatialRef()
            if srs:
                return srs

        print("Warning: CRS not found. Defaulting to EPSG:4326.")
        srs.ImportFromEPSG(4326)
        return srs

    def load_points(self, point_json, target_srs):
        with open(point_json, 'r') as f:
            data = json.load(f)

        source_srs = self.get_crs_from_geojson(point_json)
        if not source_srs:
            source_srs = osr.SpatialReference()
            source_srs.ImportFromEPSG(4326)

        transform = osr.CoordinateTransformation(source_srs, target_srs) if not source_srs.IsSame(target_srs) else None

        points = []
        values = []

        features = data.get("features", [])
        for feature in features:
            geometry = feature.get("geometry", {})
            properties = feature.get("properties", {})

            coordinates = geometry.get("coordinates", [])
            if len(coordinates) < 2:
                continue

            x, y = coordinates[:2]
            if transform:
                x, y, _ = transform.TransformPoint(x, y)
            points.append((x, y))

            value = properties.get("avg plants per square meter", None)
            values.append(np.nan if value is None else value)

        return np.array(points), np.array(values, dtype=float)

    def load_boundary(self, boundary_json):
        with open(boundary_json, 'r') as f:
            data = json.load(f)

        ring = data['features'][0]['geometry']['coordinates'][0]
        xs, ys = zip(*ring)
        return min(xs), max(xs), min(ys), max(ys)

    def idw_interpolation(self, points, values, grid_x, grid_y, power=2):
        tree = cKDTree(points)
        xi = np.array([grid_x.ravel(), grid_y.ravel()]).T

        dist, idx = tree.query(xi, k=min(5, len(points)))
        dist[dist == 0] = 1e-10

        weights = 1.0 / np.power(dist, power)
        weights /= np.sum(weights, axis=1, keepdims=True)

        interpolated = np.sum(weights * values[idx], axis=1)
        return interpolated.reshape(grid_x.shape)

    def create_raster(self, output_raster, boundary, resolution, interpolated_values, srs):
        xmin, xmax, ymin, ymax = boundary
        xres = max(1, int((xmax - xmin) / resolution))
        yres = max(1, int((ymax - ymin) / resolution))
        print(f"Raster dimensions: {xres} x {yres}")

        driver = gdal.GetDriverByName('GTiff')
        raster = driver.Create(output_raster, xres, yres, 1, gdal.GDT_Float32)
        raster.SetGeoTransform((xmin, resolution, 0, ymax, 0, -resolution))
        raster.SetProjection(srs.ExportToWkt())

        band = raster.GetRasterBand(1)
        band.WriteArray(interpolated_values[:yres, :xres])
        band.SetNoDataValue(-9999)
        raster.FlushCache()
        raster = None

        clipped_raster = output_raster.replace(".tif", "_clipped.tif")
        gdal.Warp(clipped_raster, output_raster, cutlineDSName=self.BOUNDARY_GEOJSON, cropToCutline=True, dstNodata=-9999)
        os.remove(output_raster)
        print(f"Clipped raster saved at: {clipped_raster}")

    def clip_fishnet_to_boundary(self, fishnet_file, boundary_file, output_clipped_fishnet):
        driver = ogr.GetDriverByName("GeoJSON")
        boundary_ds = driver.Open(boundary_file, 0)
        if boundary_ds is None:
            raise RuntimeError(f"Failed to open boundary file: {boundary_file}")

        boundary_layer = boundary_ds.GetLayer()
        boundary_geom = ogr.Geometry(ogr.wkbMultiPolygon)

        for feature in boundary_layer:
            geom = feature.GetGeometryRef().Clone()
            boundary_geom.AddGeometry(geom)

        fishnet_ds = driver.Open(fishnet_file, 0)
        if fishnet_ds is None:
            raise RuntimeError(f"Failed to open fishnet file: {fishnet_file}")

        fishnet_layer = fishnet_ds.GetLayer()

        clipped_ds = driver.CreateDataSource(output_clipped_fishnet)
        clipped_layer = clipped_ds.CreateLayer("clipped_fishnet", fishnet_layer.GetSpatialRef(), ogr.wkbPolygon)

        for feature in fishnet_layer:
            fishnet_geom = feature.GetGeometryRef()
            if fishnet_geom.Intersects(boundary_geom):
                clipped_feature = ogr.Feature(clipped_layer.GetLayerDefn())
                clipped_feature.SetGeometry(fishnet_geom.Intersection(boundary_geom))
                clipped_layer.CreateFeature(clipped_feature)
                clipped_feature = None

        clipped_ds = None
        fishnet_ds = None
        boundary_ds = None
        print(f"Clipped fishnet saved at: {output_clipped_fishnet}")

    def create_fishnet(self, boundary_json, output_fishnet, cell_size, srs):
        xmin, xmax, ymin, ymax = self.load_boundary(boundary_json)

        driver = ogr.GetDriverByName("GeoJSON")

        if os.path.exists(output_fishnet):
            os.remove(output_fishnet)

        fishnet_ds = driver.CreateDataSource(output_fishnet)
        if fishnet_ds is None:
            raise RuntimeError(f"Failed to create fishnet file: {output_fishnet}")

        layer = fishnet_ds.CreateLayer("fishnet", srs, ogr.wkbPolygon)

        for x in np.arange(xmin, xmax, cell_size):
            for y in np.arange(ymin, ymax, cell_size):
                ring = ogr.Geometry(ogr.wkbLinearRing)
                ring.AddPoint(x, y)
                ring.AddPoint(x + cell_size, y)
                ring.AddPoint(x + cell_size, y + cell_size)
                ring.AddPoint(x, y + cell_size)
                ring.AddPoint(x, y)

                poly = ogr.Geometry(ogr.wkbPolygon)
                poly.AddGeometry(ring)

                feature = ogr.Feature(layer.GetLayerDefn())
                feature.SetGeometry(poly)
                layer.CreateFeature(feature)
                feature = None

        fishnet_ds = None
        print(f"Fishnet saved at: {output_fishnet}")

        output_clipped_fishnet = output_fishnet.replace(".geojson", "_clipped.geojson")

        if os.path.exists(output_clipped_fishnet):
            os.remove(output_clipped_fishnet)

        self.clip_fishnet_to_boundary(output_fishnet, boundary_json, output_clipped_fishnet)

    def generate_idw_raster(self):
        srs = self.get_crs_from_geojson(self.BOUNDARY_GEOJSON)
        if not srs:
            raise ValueError("Unable to determine CRS from boundary file")

        points, values = self.load_points(self.POINTS_GEOJSON, srs)
        boundary = self.load_boundary(self.BOUNDARY_GEOJSON)

        xmin, xmax, ymin, ymax = boundary
        xres = max(1, int((xmax - xmin) / self.RESOLUTION))
        yres = max(1, int((ymax - ymin) / self.RESOLUTION))
        print(f"Grid dimensions: {xres} x {yres}")

        grid_x, grid_y = np.meshgrid(
            np.linspace(xmin, xmax, xres, endpoint=False),
            np.linspace(ymin, ymax, yres, endpoint=False)
        )

        interpolated_values = self.idw_interpolation(points, values, grid_x, grid_y)

        output_raster = os.path.join(self.OUTPUT_FOLDER, "idw_output.tif")
        self.create_raster(output_raster, boundary, self.RESOLUTION, interpolated_values, srs)

        output_fishnet = os.path.join(self.OUTPUT_FOLDER, "fishnet.geojson")
        self.create_fishnet(self.BOUNDARY_GEOJSON, output_fishnet, self.FISHNET_CELL_SIZE, srs)




class ZonalStatisticsProcessor:
    def __init__(self, raster_path, clipped_fishnet_path, target_pop, output_geojson):
        self.raster_path = raster_path
        self.clipped_fishnet_path = clipped_fishnet_path
        self.target_pop = target_pop
        self.output_geojson = output_geojson

    def extract_zonal_mean(self):
        raster_ds = gdal.Open(self.raster_path)
        if raster_ds is None:
            raise RuntimeError(f"Failed to open raster: {self.raster_path}")

        band = raster_ds.GetRasterBand(1)
        transform = raster_ds.GetGeoTransform()
        nodata_value = band.GetNoDataValue()
        raster_x_size = raster_ds.RasterXSize
        raster_y_size = raster_ds.RasterYSize

        driver = ogr.GetDriverByName("GeoJSON")
        fishnet_ds = driver.Open(self.clipped_fishnet_path, 1)
        if fishnet_ds is None:
            raise RuntimeError(f"Failed to open clipped fishnet: {self.clipped_fishnet_path}")

        layer = fishnet_ds.GetLayer()

        for field_name in ["Mean_IDW", "Percent", "class"]:
            if layer.FindFieldIndex(field_name, 1) == -1:
                field_defn = ogr.FieldDefn(field_name, ogr.OFTReal if field_name != "class" else ogr.OFTString)
                layer.CreateField(field_defn)

        features_to_delete = []
        for feature in layer:
            geometry = feature.GetGeometryRef()
            min_x, max_x, min_y, max_y = geometry.GetEnvelope()
            x_start = max(0, int((min_x - transform[0]) / transform[1]))
            x_end = min(raster_x_size, int((max_x - transform[0]) / transform[1]))
            y_start = max(0, int((max_y - transform[3]) / transform[5]))
            y_end = min(raster_y_size, int((min_y - transform[3]) / transform[5]))

            if x_end <= x_start or y_end <= y_start:
                continue

            raster_values = band.ReadAsArray(x_start, y_start, x_end - x_start, y_end - y_start)
            if raster_values is None:
                continue

            valid_values = raster_values[raster_values != nodata_value]
            mean_value = float(np.mean(valid_values)) if valid_values.size > 0 else -9999

            percent = (mean_value / self.target_pop) * 100 if mean_value != -9999 else -9999
            if percent >= 90:
                class_value = 3
            elif 80 <= percent < 90:
                class_value = 2
            elif 70 <= percent < 80:
                class_value = 1
            elif percent < 70:
                class_value = 0
            else:
                features_to_delete.append(feature.GetFID())
                continue

            feature.SetField("Mean_IDW", mean_value)
            feature.SetField("Percent", percent)
        
            feature.SetField("class", class_value)
            layer.SetFeature(feature)

        for fid in reversed(features_to_delete):
            layer.DeleteFeature(fid)

        fishnet_ds = None
        raster_ds = None
        print(f"Attributes updated in {self.clipped_fishnet_path}")

    def dissolve_by_class(self):
        driver = ogr.GetDriverByName("GeoJSON")
        input_ds = driver.Open(self.clipped_fishnet_path, 0)
        if input_ds is None:
            raise RuntimeError(f"Failed to open input file: {self.clipped_fishnet_path}")

        input_layer = input_ds.GetLayer()
        spatial_ref = input_layer.GetSpatialRef()

        dissolved_ds = driver.CreateDataSource(self.output_geojson)
        dissolved_layer = dissolved_ds.CreateLayer("dissolved", spatial_ref, ogr.wkbPolygon)

        class_field = ogr.FieldDefn("class", ogr.OFTInteger)
        percent_field = ogr.FieldDefn("AreaPercent", ogr.OFTReal)
        area_field = ogr.FieldDefn("Area", ogr.OFTReal)
        dissolved_layer.CreateField(class_field)
        dissolved_layer.CreateField(percent_field)
        dissolved_layer.CreateField(area_field)

        class_dict = {}
        total_area = 0

        for feature in input_layer:
            geom = feature.GetGeometryRef().Clone()
            class_value = feature.GetField("class")
            if class_value is None:
                continue

            area = geom.GetArea()
            total_area += area

            if class_value in class_dict:
                class_dict[class_value]["geom"] = class_dict[class_value]["geom"].Union(geom)
                class_dict[class_value]["area"] += area
            else:
                class_dict[class_value] = {"geom": geom, "area": area}

        for class_value, data in class_dict.items():
            new_feature = ogr.Feature(dissolved_layer.GetLayerDefn())
            new_feature.SetGeometry(data["geom"])
            new_feature.SetField("class", class_value)
            new_feature.SetField("Area", data["area"])
            new_feature.SetField("AreaPercent", (data["area"] / total_area) * 100)
            dissolved_layer.CreateFeature(new_feature)

        input_ds = None
        dissolved_ds = None
        print(f"Dissolved polygons saved to {self.output_geojson}")

