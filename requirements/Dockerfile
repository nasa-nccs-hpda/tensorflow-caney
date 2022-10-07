# Arguments to pass to the image
ARG TF_VER=tf2
ARG VERSION_DATE=22.08
ARG FROM_IMAGE=nvcr.io/nvidia/tensorflow

# Import RAPIDS container as the BASE Image (cuda base image)
FROM ${FROM_IMAGE}:${VERSION_DATE}-${TF_VER}-py3

# Ubuntu needs noninteractive to be forced
ENV DEBIAN_FRONTEND noninteractive
ENV LD_LIBRARY_PATH="/mnt/data-store/FileGDB_API-64gcc51/lib/:/usr/local/lib"
ENV PROJ_LIB="/usr/share/proj"
ENV CPLUS_INCLUDE_PATH="/usr/include/gdal"
ENV C_INCLUDE_PATH="/usr/include/gdal"

# system dependencies
RUN apt-get update && \
    apt-get -y install software-properties-common && \
    add-apt-repository ppa:ubuntugis/ppa && \
    apt-get update && apt-get -y dist-upgrade && \
    apt-get -y install build-essential libsm6 libxext6 libxrender-dev libfontconfig1 \
        bzip2 diffutils file build-essential make swig wget libproj-dev libnetcdf-dev \
        gdal-bin libgdal-dev && \
    apt-get -y autoremove && \
    rm -rf /var/cache/apt

# Pip
RUN pip --no-cache-dir install omegaconf rasterio rioxarray xarray geopandas tensorflow_probability \
    opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless \
    tifffile webcolors Pillow seaborn xgboost segmentation-models pytest coveralls \
    GDAL==`ogrinfo --version | grep -Eo '[0-9]\.[0-9]\.[0-9]+'`

HEALTHCHECK NONE
ENTRYPOINT []
CMD ["/bin/bash"]
