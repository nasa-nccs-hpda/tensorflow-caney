import rioxarray as rxr
from geospatial_file import GeospatialFile


class VegetationIndices(GeospatialFile):

    def __init__(self, filename, indices_list):

        self.indices_list = indices_list
        self.filename = filename
        self.data = rxr.open_rasterio(filename)

    def _ndvi():
        print("Calculate ndvi here")


if __name__ == "__main__":

    raster = VegetationIndices(
        '2013_NEON_D17_SJER_DP3_257000_4105000_reflectance.h5',
        ['ndvi']
    )
    print(len(raster.data))
    print(raster.indices_list)
