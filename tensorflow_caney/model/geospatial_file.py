import rioxarray as rxr

# singularity shell --nv -B \
# /lscratch,/css,/explore/nobackup/projects/ilab,/explore/nobackup/people \
# /explore/nobackup/projects/ilab/containers/tensorflow-caney-2022.12


class GeospatialFile(object):

    def __init__(self, filename):

        self.filename = filename
        self.data = rxr.open_rasterio(filename)

    # -------------------------------------------------------------------------
    # fileName
    # -------------------------------------------------------------------------
    """
    def modify_bands(
            xraster: xr.core.dataarray.DataArray, input_bands: List[str],
            output_bands: List[str], drop_bands: List[str] = []):
        Drop multiple bands to existing rasterio object
        # Drop any bands from input that should not be on output
        for ind_id in list(set(input_bands) - set(output_bands)):
            drop_bands.append(input_bands.index(ind_id)+1)

        if isinstance(xraster, (np.ndarray, np.generic)):
            # Do not modify if image has the same number of output bands
            if xraster.shape[-1] == len(output_bands):
                return xraster
            xraster = np.delete(
                xraster[:][:], [x - 1 for x in drop_bands], axis=0)
            return xraster
        else:
            # Do not modify if image has the same number of output bands
            if xraster['band'].shape[0] == len(output_bands):
                return xraster
            return xraster.drop(dim="band", labels=drop_bands)
    """


if __name__ == "__main__":

    raster = GeospatialFile(
        '2013_NEON_D17_SJER_DP3_257000_4105000_reflectance.h5')
    print(raster.data[0].shape)
