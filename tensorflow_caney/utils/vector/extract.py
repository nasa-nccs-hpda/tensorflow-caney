import os
import re
import logging
import osgeo.gdal
import numpy as np
import pandas as pd
import geopandas as gpd
from glob import glob
from pathlib import Path
from skimage.draw import polygon
from skimage.filters import threshold_mean


def filter_gdf_by_list(
            gdf: gpd.GeoDataFrame,
            gdf_key: str = 'acq_year',
            isin_list: list = [],
            reset_index: bool = True
        ):
    """
    Filter GDF by list of values.
    Args:
        gdf (gpd.GeoDataFrame): geodataframe to modify
        gdf_key (str): column key to perform operations on
        isin_list (list): list of features to filter on
        reset_index (bool): reset geodataframe index
    Returns:
        gpd.GeoDataFrame
    """
    return gdf[gdf[gdf_key].isin(isin_list)].reset_index(drop=reset_index)


def get_polygon_total_bounds(
            data_regex: str,
            date_column_name: str = 'acq_year',
            drop_column_names: list = ["DN", "PXLVAL"]
        ) -> gpd.DataFrame:
    """
    Generate geodataframe from a list of GPKGs or shapefiles
    and take the geometries of the polygons as a geometry object.
    Args:
        data_regex (str): regex string to glob data
        date_column_name (str): column name for timestamp features
        drop_column_names (list): list of features to drop
    Returns:
        gpd.GeoDataFrame
    """
    # get the paths/filenames of the world view imagery available
    filenames = []
    if isinstance(data_regex, list):
        for regex in data_regex:
            filenames.extend(glob(regex))
    else:
        filenames = glob(regex)

    # iterate over all filenames, and create list of geodataframes
    polygon_gdf_list = []
    for filename in filenames:

        # get year string
        year_id = re.search(r'\d{4}', filename)

        # read filename, get date, and drop unwanted
        polygon_gdf = gpd.read_file(filename)
        polygon_gdf[date_column_name] = int(year_id.group(0))
        polygon_gdf = polygon_gdf.drop(
            drop_column_names, axis=1, errors='ignore')
        polygon_gdf_list.append(polygon_gdf)
    return pd.concat(polygon_gdf_list)


def extract_window(
            image_dataset,
            pixel_x: int,
            pixel_y: int,
            pixel_width: int,
            pixel_height: int
        ) -> np.ndarray:
    """
    Extract window from raster or from numpy array.
    Args:
        image_dataset (gdal.Dataset): gdal dataset to extract window
        pixel_x (int): integer for pixel position on x
        pixel_y (int): integer for pixel position on y
        pixel_width (int): integer for pixel width
        pixel_height (int): integer for pixel height
    Returns:
        np.ndarray
    """
    if type(image_dataset) is np.ndarray:
        data = image_dataset[
            int(pixel_x):int(pixel_x+pixel_width),
            int(pixel_y):int(pixel_y+pixel_height)]
    else:
        data = image_dataset.ReadAsArray(
            pixel_x, pixel_y, pixel_width, pixel_height)
    return data


def extract_centered_window(
            image_dataset,
            pixel_x: int,
            pixel_y: int,
            pixel_width: int,
            pixel_height: int
        ):
    """
    Define center pixels and call window extraction.
    Args:
        image_dataset (gdal.Dataset): gdal dataset to extract window
        pixel_x (int): integer for pixel position on x
        pixel_y (int): integer for pixel position on y
        pixel_width (int): integer for pixel width
        pixel_height (int): integer for pixel height
    Returns:
        np.ndarray
    """
    centered_pixel_x = pixel_x - pixel_width / 2
    centered_pixel_y = pixel_y - pixel_height / 2
    return extract_window(
        image_dataset, centered_pixel_x, centered_pixel_y,
        pixel_width, pixel_height
    )


def convert_coords_to_pixel_location(coords: list, transform: list = None):
    """
    Convert geographical coordinates to array pixel location.
    Args:
        coords (list): coordinate from the rater location, [x, y]
        transform (list): transform list from gdal transform
    Returns:
        (x coordinate, y coordinate)
    """
    x_geo, y_geo = coords[0], coords[1]
    g0, g1, g2, g3, g4, g5 = transform
    if g2 == 0:
        x_pixel = (x_geo - g0) / float(g1)
        y_pixel = (y_geo - g3 - x_pixel * g4) / float(g5)
    else:
        x_pixel = (y_geo * g2 - x_geo * g5 + g0 * g5 - g2 * g3) \
            / float(g2 * g4 - g1 * g5)
        y_pixel = (x_geo - g0 - x_pixel * g1) / float(g2)
    return int(round(x_pixel)), int(round(y_pixel))


def extract_tiles(gpd_iter: gpd.GeoDataframe):
    """
    Extract and save tile from pandas dataframe metadata.
    Args:
        gpd_iter (gpd.GeoDataframe): row from geodataframe
    Returns:
        None (saves np.array to disk)
    """
    try:
        # decompress row iter object into row_id and geopandas row
        row_id, row = gpd_iter

        # set output filenames
        output_data_filename = os.path.join(
            row['images_output_dir'],
            f'{Path(row["scene_id"]).stem}_{str(row_id+1)}.npy'
        )
        output_label_filename = os.path.join(
            row['labels_output_dir'],
            f'{Path(row["scene_id"]).stem}_{str(row_id+1)}.npy'
        )

        # open raster via GDAL
        image_dataset = osgeo.gdal.Open(row['scene_id'])

        # get window pixel location from coordinates
        window_pixel_x, window_pixel_y = convert_coords_to_pixel_location(
            [row['geometry'].centroid.x, row['geometry'].centroid.y],
            image_dataset.GetGeoTransform()
        )

        # extract data array from the greater raster
        data_array = extract_centered_window(
            image_dataset, window_pixel_x, window_pixel_y,
            row['tile_size'], row['tile_size']
        )

        # if nodata is present, or if tiles is incorrect, skip
        if data_array is None or data_array.min() < 0:
            return

        # move axis to be channels last
        data_array = np.moveaxis(data_array, 0, -1)

        # here we extract the label tile
        label_mask = np.full(
            (image_dataset.RasterXSize, image_dataset.RasterYSize), False)

        # get polygon pixel coordinates
        polygon_pixel_coords = np.apply_along_axis(
            convert_coords_to_pixel_location,
            axis=1,
            arr=np.array(list(row['geometry'].exterior.coords)),
            transform=image_dataset.GetGeoTransform()
        )

        # extract a polygon object, and mask out the location of the polygon
        rr, cc = polygon(
            polygon_pixel_coords[:, 0], polygon_pixel_coords[:, 1],
            (image_dataset.RasterXSize, image_dataset.RasterYSize)
        )
        label_mask[cc, rr] = True

        # there is a strange represenation of where X and Y are location within
        # the polygon for now just invert the axis and call it a day
        label_array = extract_centered_window(
            label_mask.astype(int), window_pixel_y, window_pixel_x,
            row['tile_size'], row['tile_size']
        )

        # ignore tiles that do not have true occurrences
        if np.count_nonzero(label_array == 1) < 15000:
            return

        # assert if a tile is smaller than the given size we want
        if label_array.shape[0] != row['tile_size'] or \
                label_array.shape[1] != row['tile_size']:
            return

        # do segmentation with threshold to improve the appearance
        # given the difference in resolution from raster and label
        label_array = data_array[:, :, row['clustering_band']] > \
            threshold_mean(
                data_array[:, :, row['clustering_band']]
            )
        label_array = np.expand_dims(label_array, axis=-1)

        # output to disk
        np.save(output_data_filename, data_array)
        np.save(output_label_filename, label_array)

        # extracting none-crop data arrays
        data_array = extract_centered_window(
            image_dataset, window_pixel_x + 1000, window_pixel_y + 1000,
            row['tile_size'], row['tile_size']
        )

        # if nodata is present, or if tiles is incorrect, skip
        if data_array is None or data_array.min() < 0:
            return

        # move axis to be channels last
        data_array = np.moveaxis(data_array, 0, -1)
        label_array = np.zeros((256, 256))
        output_data_filename = os.path.join(
            row['images_output_dir'],
            f'{Path(row["scene_id"]).stem}_{str(row_id+1)}_nocrop.npy'
        )
        output_label_filename = os.path.join(
            row['labels_output_dir'],
            f'{Path(row["scene_id"]).stem}_{str(row_id+1)}_nocrop.npy'
        )
        # output to disk
        np.save(output_data_filename, data_array)
        np.save(output_label_filename, label_array)

    except (AttributeError, IndexError) as e:
        logging.info(e)
        return
    return
