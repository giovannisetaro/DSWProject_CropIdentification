
from datetime import datetime, timedelta
import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import shape
import ee 
import os
import math

def create_gee_geometries(coords_list, side_km=10):
    """
    Creates square geometries of side `side_km` km (default 10 km) centered on each coordinate.

    Args:
        coords_list (list of tuples): list of (lon, lat) coordinates in degrees (EPSG:4326).
        side_km (float): side length of the square in kilometers (default 10 km).

    Returns:
        list of ee.Geometry.Polygon: list of GEE polygons.
    """
    # Approximate conversion km -> degrees latitude (1 deg latitude ~111 km)
    side_deg_lat = side_km / 111.0

    geometries = []
    for lon, lat in coords_list:
        # Longitude degrees per km varies with latitude
        side_deg_lon = side_km / (111.320 * abs(math.cos(math.radians(lat))) + 1e-9)

        half_lon = side_deg_lon / 2
        half_lat = side_deg_lat / 2

        # Polygon corners (closed polygon)
        polygon_coords = [
            [lon - half_lon, lat - half_lat],
            [lon - half_lon, lat + half_lat],
            [lon + half_lon, lat + half_lat],
            [lon + half_lon, lat - half_lat],
            [lon - half_lon, lat - half_lat],
        ]

        geom = ee.Geometry.Polygon(polygon_coords)
        geometries.append(geom)

    return geometries






def generate_half_month_ranges(start_date, end_date):
    date_list = []
    current = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    while current < end:
        mid = current + timedelta(days=15)
        end_of_month = (current.replace(day=28) + timedelta(days=4)).replace(day=1)

        date_list.append((current.strftime('%Y-%m-%d'), min(mid, end).strftime('%Y-%m-%d')))
        date_list.append((min(mid, end).strftime('%Y-%m-%d'), min(end_of_month, end).strftime('%Y-%m-%d')))

        current = end_of_month

    return date_list

def generate_date_ranges(start_date, end_date, interval_days=15):
    """
    Génère des intervalles de dates de `interval_days` jours entre `start_date` et `end_date`.

    Args:
        start_date (str): Date de début au format 'YYYY-MM-DD'.
        end_date (str): Date de fin au format 'YYYY-MM-DD'.
        interval_days (int): Longueur de chaque intervalle en jours.

    Returns:
        List[Tuple[str, str]]: Liste de tuples (date_début, date_fin).
    """
    date_list = []
    current = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    while current < end:
        next_date = current + timedelta(days=interval_days)
        date_list.append((
            current.strftime('%Y-%m-%d'),
            min(next_date, end).strftime('%Y-%m-%d')
        ))
        current = next_date

    return date_list









def mask_scl(img):
    scl = img.select('SCL')
    # Mask all pixels that are NOT in classes 4 to 7
    mask = (scl.eq(4)
           .Or(scl.eq(5))
           .Or(scl.eq(6))
           .Or(scl.eq(7)))
    return img.updateMask(mask)

def get_median_image(start, end, Geometry_data_collect):
    # Load the useful bands + SCL
    collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
        .filterBounds(Geometry_data_collect) \
        .filterDate(start, end) \
        .map(lambda img: img.clip(Geometry_data_collect)) \
        .map(mask_scl) \
        .select(bands)

    def compute_median():
        return collection.median() \
            .set('system:time_start', ee.Date(start).millis()) \
            .set('start_date', start) \
            .set('end_date', end) \
            .set('Number_of_aggreted_images', size)

    size = collection.size()
    median = ee.Algorithms.If(size.gt(0), compute_median(), None)
    return median


def export_tiff(image_collection,Geometry_data_collect,export_folder,export_maxPixels ): 
    # pixel overlap

    image_list = image_collection.toList(image_collection.size())
    n = image_collection.size().getInfo()
    
    for i in range(n):
        image = ee.Image(image_list.get(i)).clip(Geometry_data_collect)
        
        # Build a unique name, for example acquisition date
        date_str = image.date().format('YYYYMMdd').getInfo()
        task = ee.batch.Export.image.toDrive(
            image=image,
            description= "export",
            folder=export_folder,
            fileNamePrefix=f'sentinel2_{date_str}',
            region=Geometry_data_collect,
            scale=10,
            crs='EPSG:4326',
            maxPixels=export_maxPixels
        )
        task.start()
        print(f"Export started for image {i+1}/{n}, date : {date_str}")


        import os
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
from shapely.geometry import box

def rasterize_labels_per_zone(
    base_dir,
    gdf,
    label_column="LABEL",
    tif_suffix_filter=".tif",
    label_filename="labels_raster.tif"
):
    """
    Rasterizes the polygons from the GeoDataFrame for each zone found in subfolders.

    Args:
        base_dir (str): Path to the main directory containing one subfolder per zone.
        gdf (GeoDataFrame): GeoDataFrame with polygon geometries and a class column.
        label_column (str): Name of the column in `gdf` containing class IDs.
        tif_suffix_filter (str): File extension filter to locate reference .tif images.
        label_filename (str): Name of the output label raster file to save in each zone folder.

    Returns:
        None
    """

    for zone_name in os.listdir(base_dir):
        zone_path = os.path.join(base_dir, zone_name)
        if not os.path.isdir(zone_path):
            continue

        print(f"🔄  processing zone n°: {zone_name}")

        # Trouver une image de référence dans le sous-dossier
        tif_files = sorted([f for f in os.listdir(zone_path) if f.endswith(tif_suffix_filter)])
        if not tif_files:
            print(f"⚠️ no .tif founded in {zone_path}")
            continue

        ref_path = os.path.join(zone_path, tif_files[0])

        with rasterio.open(ref_path) as src:
            meta = src.meta.copy()
            bounds = src.bounds
            transform = src.transform

        # Filtrer les polygones qui intersectent l'image de référence
        img_bbox = box(*bounds)
        gdf_zone = gdf[gdf.geometry.intersects(img_bbox)]

        if gdf_zone.empty:
            print(f"⚠️ no field polygones founded in  {zone_name}.")
            continue

        # Rasterisation
        shapes = ((geom, value) for geom, value in zip(gdf_zone.geometry, gdf_zone[label_column]))
        label_raster = rasterize(
            shapes,
            out_shape=(meta['height'], meta['width']),
            transform=transform,
            fill=0,
            dtype=rasterio.uint8
        )

        # Écriture du fichier raster labelisé
        meta.update({'count': 1, 'dtype': rasterio.uint8})
        label_path = os.path.join(zone_path, label_filename)

        with rasterio.open(label_path, 'w', **meta) as dst:
            dst.write(label_raster, 1)

        print(f"✅ ground trouth label saved : {label_path}")
