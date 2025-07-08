
from datetime import datetime, timedelta
import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import shape
import ee 
import os
import math
from tqdm import tqdm
import numpy as np
import os
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
from shapely.geometry import box

from dateutil.relativedelta import relativedelta

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



def generate_monthly_ranges(start_date, end_date):
    """
    Génère une liste de tuples représentant les intervalles mensuels
    entre start_date et end_date (inclus).
    """
    date_list = []
    current = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    while current < end:
        next_month = current + relativedelta(months=1)
        date_list.append((
            current.strftime('%Y-%m-%d'),
            min(next_month, end).strftime('%Y-%m-%d')
        ))
        current = next_month

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
    bands = ['B2', 'B3', 'B4', 'B8']
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

    
    n_images = median.size().getInfo()
    median_list = median.toList(median.size())

    total_n_img = 0
    print(f"median images for date {start} retrieved ! infos : ")
    for i in range(n_images):
        img = ee.Image(median_list.get(i))
        props = img.toDictionary().getInfo()
        num_agg = props['Number_of_aggreted_images']
        total_n_img += num_agg
        print(f"Image {i} composed by {num_agg} images " )

    print(total_n_img)

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

        #  Rasterisation of classes
        shapes_label = ((geom, value) for geom, value in zip(gdf_zone.geometry, gdf_zone[label_column]))
        label_raster = rasterize(
            shapes_label,
            out_shape=(meta['height'], meta['width']),
            transform=transform,
            fill=0,
            dtype=rasterio.uint8
        )

        # 4. Rasterisation of ID_PARCEL 
        shapes_id = ((geom, int(pid)) for geom, pid in zip(gdf_zone.geometry, gdf_zone["ID_PARCEL"]))
        id_raster = rasterize(
            shapes_id,
            out_shape=(meta['height'], meta['width']),
            transform=transform,
            fill=0,
            dtype=rasterio.uint32
        )

        # 5. Mise à jour des métadonnées
        meta.update({
            'count': 2,                     # 2 couches : label + ID_PARCEL
            'dtype': 'uint32'              # pour être sûr que ID_PARCEL soit stocké correctement
        })

        # 6. Sauvegarde dans le fichier
        label_path = os.path.join(zone_path, label_filename)

        with rasterio.open(label_path, 'w', **meta) as dst:
            dst.write(label_raster.astype('uint32'), 1)  # couche 1 : labels
            dst.write(id_raster, 2)                      # couche 2 : ID_PARCEL

        print(f"✅ ground trouth label saved : {label_path}")



def mask_no_labeled_pixel_all_zones(base_dir, label_filename="labels_raster.tif", delete_original=True):
    """
    Iterates over all subfolders in `base_dir`, reads the label raster file in each zone,
    then masks all pixels with label value 0 on every TIFF file within that zone.

    Args:
        base_dir (str): Path to the main directory containing zone subfolders.
        label_filename (str): Name of the label raster file within each zone folder.
        delete_original (bool): If True, deletes the original TIFF files after masking.

    Returns:
        None
    """
    for zone_name in os.listdir(base_dir):
        zone_path = os.path.join(base_dir, zone_name)
        if not os.path.isdir(zone_path):
            continue
        
        label_path = os.path.join(zone_path, label_filename)
        if not os.path.exists(label_path):
            print(f"⚠️ Label raster not found in {zone_path}, skipping.")
            continue
        
        print(f"🔄 Masking TIFFs in zone: {zone_name}")
        
        with rasterio.open(label_path) as src:
            img_reference = src.read(1)
        
        tif_files = sorted([
            os.path.join(zone_path, f)
            for f in os.listdir(zone_path)
            if f.endswith('.tif') and not f.endswith('_masked.tif')
        ])
        
        for tif_path in tqdm(tif_files, desc=f"Masking TIFFs in {zone_name}"):
            with rasterio.open(tif_path) as src:
                data = src.read()
                profile = src.profile
                
                mask = (img_reference == 0)
                data[:, mask] = 0
            
            base_name = os.path.basename(tif_path).replace('.tif', '_masked.tif')
            output_path = os.path.join(zone_path, base_name)
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(data)
            
            if delete_original:
                os.remove(tif_path)
        
        print(f"✅ Masking done for zone {zone_name}")


def count_total_nan_in_tifs(base_dir="../data/Tif"):
    total_nan_global = 0

    for zone in sorted(os.listdir(base_dir)):
        zone_path = os.path.join(base_dir, zone)
        if not os.path.isdir(zone_path):
            continue
        total_nan_zone = 0
        print(f"Traitement de la zone : {zone}")

        for fname in os.listdir(zone_path):
            if fname.endswith('.tif') and 'labels' not in fname:
                fpath = os.path.join(zone_path, fname)
                with rasterio.open(fpath) as src:
                    img = src.read(1)
                    nan_count = np.isnan(img).sum()
                    total_nan_zone += nan_count
        print(f"  Nombre total de NaN dans {zone}: {total_nan_zone}")
        total_nan_global += total_nan_zone

    print(f"Nombre total de NaN dans toutes les zones : {total_nan_global}")
    return total_nan_global