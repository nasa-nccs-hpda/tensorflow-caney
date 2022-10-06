import os
import rasterio as rio
import geopandas as gpd
from glob import glob
from pathlib import Path
from shapely.geometry import box


def get_raster_total_bounds(
            data_regex: str,
            crs: str = None,
            date_column_name: str = 'acq_year'
        ) -> gpd.GeoDataFrame:
    """
    Generate geodataframe from a list of GeoTIFFs and take the
    external boundaries of the GeoTIFFs as a geometry object.
    Args:
        data_regex (str): regex string to glob data
        crs (str): crs to convert geodataframe to
        date_column_name (str): column name for timestamp features
    Returns:
        gpd.GeoDataFrame
    """
    # get the paths/filenames of the world view imagery available
    filenames = []
    if isinstance(data_regex, list):
        for regex in data_regex:
            filenames.extend(glob(regex))
    else:
        filenames = glob(data_regex)

    # define variables to store the output of the searches
    scene_ids_list, bounds_ids_list, years_ids_list, \
        study_area_list = [], [], [], []

    # if not CRS is given, its taken from the first GeoTIFF
    if crs is None:
        crs = rio.open(filenames[0]).crs

    for filename in filenames:

        # append some metadata
        scene_ids_list.append(filename)
        years_ids_list.append(int(Path(filename).stem[5:9]))
        bounds_ids_list.append(box(*rio.open(filename).bounds))

        # adding site name
        if os.path.basename(os.path.dirname(filename)) == 'M1BS':
            study_area_list.append(os.path.basename(
                os.path.dirname(os.path.dirname(filename))))
        else:
            study_area_list.append(os.path.basename(
                os.path.dirname(filename)))

    df_metadata = {
        'study_area': study_area_list,
        'scene_id': scene_ids_list,
        date_column_name: years_ids_list,
        'geometry': bounds_ids_list
    }
    return gpd.GeoDataFrame(df_metadata, crs=crs)
