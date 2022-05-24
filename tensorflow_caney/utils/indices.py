import xarray as xr  # read rasters
import numpy as np

try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

__author__ = "Jordan A Caraballo-Vega, Science Data Processing Branch"
__email__ = "jordan.a.caraballo-vega@nasa.gov"
__status__ = "Development"

# -------------------------------------------------------------------------------
# module indices
# This class calculates remote sensing indices given xarray or numpy objects.
# Note: Most of our imagery uses the following set of bands.
# 8 band: ['CoastalBlue', 'Blue', 'Green', 'Yellow',
#          'Red', 'RedEdge', 'NIR1', 'NIR2']
# 4 band: ['Red', 'Green', 'Blue', 'NIR1', 'HOM1', 'HOM2']
# -------------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Module Methods
# ----------------------------------------------------------------------------

__all__ = [
    "cs1", "cs2", "dvi", "dwi", "fdi", "ndvi", "ndwi",
    "si", "_get_band_locations", "get_indices", "add_indices"
]

# ---------------------------------------------------------------------------
# Get Methods
# ---------------------------------------------------------------------------
def _get_band_locations(bands: list, requested_bands: list):
    """
    Get list indices for band locations.
    """
    locations = []
    for b in requested_bands:
        try:
            locations.append(bands.index(b.lower()))
        except ValueError:
            raise ValueError(f'{b} not in raster bands {bands}')
    return locations

# ---------------------------------------------------------------------------
# Indices Methods
# ---------------------------------------------------------------------------
def cs1(raster):
    """
    Cloud detection index (CS1), CS1 := (3. * NIR1) / (Blue + Green + Red)
    :param raster: xarray or numpy array object in the form (c, h, w)
    :return: new band with SI calculated
    """
    nir1, red, blue, green = _get_band_locations(
        raster.attrs['band_names'], ['nir1', 'red', 'blue', 'green'])
    index = (
        (3. * raster[nir1, :, :]) /
        (raster[blue, :, :] + raster[green, :, :] \
            + raster[red, :, :])
    )
    return index.expand_dims(dim="band", axis=0)


def cs2(raster):
    """
    Cloud detection index (CS2), CS2 := (Blue + Green + Red + NIR1) / 4.
    :param raster: xarray or numpy array object in the form (c, h, w)
    :return: new band with CS2 calculated
    """
    nir1, red, blue, green = _get_band_locations(
        raster.attrs['band_names'], ['nir1', 'red', 'blue', 'green'])
    index = (
        (raster[blue, :, :] + raster[green, :, :] \
            + raster[red, :, :] + raster[nir1, :, :])
        / 4.0
    )
    return index.expand_dims(dim="band", axis=0)


def dvi(raster):
    """
    Difference Vegetation Index (DVI), DVI := NIR1 - Red
    :param raster: xarray or numpy array object in the form (c, h, w)
    :return: new band with DVI calculated
    """
    nir1, red = _get_band_locations(
        raster.attrs['band_names'], ['nir1', 'red'])
    index = (
        raster[nir1, :, :] - raster[red, :, :]
    )
    return index.expand_dims(dim="band", axis=0)


def dwi(raster):
    """
    Difference Water Index (DWI), DWI := Green - NIR1
    :param raster: xarray or numpy array object in the form (c, h, w)
    :return: new band with DWI calculated
    """
    nir1, green = _get_band_locations(
        raster.attrs['band_names'], ['nir1', 'green'])
    index = (
        raster[green, :, :] - raster[nir1, :, :]
    )
    return index.expand_dims(dim="band", axis=0)


def fdi(raster):
    """
    Forest Discrimination Index (FDI), type int16
    8 band imagery: FDI := NIR2 - (RedEdge + Blue)
    4 band imagery: FDI := NIR1 - (Red + Blue)
    :param data: xarray or numpy array object in the form (c, h, w)
    :return: new band with FDI calculated
    """
    bands = ['blue', 'nir2', 'rededge']
    if not all(b in bands for b in raster.attrs['band_names']):
        bands = ['blue', 'nir1', 'red']
    blue, nir, red = _get_band_locations(
        raster.attrs['band_names'], bands)
    index = (
        raster[nir, :, :] - \
            (raster[red, :, :] + raster[blue, :, :])
    )
    return index.expand_dims(dim="band", axis=0)


def gndvi(raster):
    """
    Difference Vegetation Index (DVI), GNDVI := (NIR - Green) / (NIR + Green)
    :param raster: xarray or numpy array object in the form (c, h, w)
    :return: new band with DVI calculated
    """
    nir1, green = _get_band_locations(
        raster.attrs['band_names'], ['nir1', 'green'])
    index = (
        (raster[nir1, :, :] - raster[green, :, :]) /
        (raster[nir1, :, :] + raster[green, :, :])
    )
    return index.expand_dims(dim="band", axis=0)


def ndvi(raster):
    """
    Difference Vegetation Index (DVI), NDVI := (NIR - Red) / (NIR + RED)
    :param raster: xarray or numpy array object in the form (c, h, w)
    :return: new band with DVI calculated
    """
    nir1, red = _get_band_locations(
        raster.attrs['band_names'], ['nir1', 'red'])
    index = (
        (raster[nir1, :, :] - raster[red, :, :]) /
        (raster[nir1, :, :] + raster[red, :, :])
    )
    return index.expand_dims(dim="band", axis=0)


def ndwi(raster):
    """
    Normalized Difference Water Index (NDWI)
    NDWI := factor * (Green - NIR1) / (Green + NIR1)
    :param raster: xarray or numpy array object in the form (c, h, w)
    :return: new band with SI calculated
    """
    nir1, green = _get_band_locations(
        raster.attrs['band_names'], ['nir1', 'green'])
    index = (
        (raster[green, :, :] - raster[nir1, :, :]) /
        (raster[green, :, :] + raster[nir1, :, :])
    )
    return index.expand_dims(dim="band", axis=0)


def si(raster):
    """
    Shadow Index (SI), SI := (Blue * Green * Red) ** (1.0 / 3)
    :param raster: xarray or numpy array object in the form (c, h, w)
    :return: new band with SI calculated
    """
    red, blue, green = _get_band_locations(
        raster.attrs['band_names'], ['red', 'blue', 'green'])
    index = (
        (raster[blue, :, :] - raster[green, :, :] /
            raster[red, :, :]) ** (1.0/3.0)
    )
    return index.expand_dims(dim="band", axis=0)


def sr(raster):
    """
    SR := NIR / Red
    :param raster: xarray or numpy array object in the form (c, h, w)
    :return: new band with SI calculated
    """
    nir1, red = _get_band_locations(
        raster.attrs['band_names'], ['nir1', 'red'])
    index = (
        raster[nir1, :, :] / raster[red, :, :]
    )
    return index.expand_dims(dim="band", axis=0)


# ---------------------------------------------------------------------------
# Modify Methods
# ---------------------------------------------------------------------------
indices_registry = {
    'cs1': cs1,
    'cs2': cs2,
    'dvi': dvi,
    'dwi': dwi,
    'fdi': fdi,
    'gndvi': gndvi,
    'ndvi': ndvi,
    'ndwi': ndwi,
    'si': si,
    'sr': sr
}

def _get_index_function(index_key):
    """
    Get index function from the indices registry.
    """
    try:
        return indices_registry[index_key.lower()]
    except KeyError:
        raise ValueError(f'Invalid indices mapping: {index_key}.')

def add_indices(xraster, input_bands, output_bands, factor=1.0):
    """
    :param rastarr: xarray or numpy array object in the form (c, h, w)
    :param bands: list with strings of bands in the raster
    :param indices: indices to calculate and append to the raster
    :param factor: factor used for toa imagery
    :return: raster with updated bands list
    """
    n_bands = len(input_bands)  # get number of input bands

    # make band names uniform for easy of validation
    input_bands = [s.lower() for s in input_bands]
    output_bands = [s.lower() for s in output_bands]
    
    # add an attribute to the raster
    xraster.attrs['band_names'] = input_bands

    # iterate over each new band that needs to be included
    for band_id in output_bands:

        if band_id not in input_bands:

            # increase the number of input bands
            n_bands += 1

            # calculate band (indices)
            index_function = _get_index_function(band_id)
            new_index = index_function(xraster)

            # Add band indices to raster, add future object
            new_index.coords['band'] = [n_bands]
            xraster = xr.concat([xraster, new_index], dim='band')

            # Update raster metadata,
            xraster.attrs['band_names'].append(band_id)

    return xraster
