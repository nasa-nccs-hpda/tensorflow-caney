# Arguments to pass to the image
ARG TF_VER=tf2
ARG VERSION_DATE=22.08
ARG FROM_IMAGE=nvcr.io/nvidia/tensorflow

FROM osgeo/gdal:ubuntu-full-latest as builder

# Import RAPIDS container as the BASE Image (cuda base image)
FROM ${FROM_IMAGE}:${VERSION_DATE}-${TF_VER}-py3

# Ubuntu needs noninteractive to be forced
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get -y dist-upgrade && \
    apt-get -y install build-essential libsm6 libxext6 libxrender-dev libfontconfig1 && \
    apt-get -y autoremove && \
    rm -rf /var/cache/apt

# Pip
RUN pip --no-cache-dir install omegaconf rasterio rioxarray xarray geopandas \
    opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless \
    tifffile webcolors Pillow seaborn xgboost segmentation-models
    #pip uninstall -y cupy-cuda114 && pip install cupy-cuda115

ARG PROJ_INSTALL_PREFIX=/usr/local
COPY --from=builder  ${PROJ_INSTALL_PREFIX}/share/proj ${PROJ_INSTALL_PREFIX}/share/proj
COPY --from=builder  ${PROJ_INSTALL_PREFIX}/include/* ${PROJ_INSTALL_PREFIX}/include
COPY --from=builder  ${PROJ_INSTALL_PREFIX}/bin/ ${PROJ_INSTALL_PREFIX}/bin/
COPY --from=builder  ${PROJ_INSTALL_PREFIX}/lib/lib* ${PROJ_INSTALL_PREFIX}/lib
COPY --from=builder  /usr/share/java /usr/share/java
COPY --from=builder  /usr/share/gdal/ /usr/share/gdal/
COPY --from=builder  /usr/include/ /usr/include/
COPY --from=builder  /usr/ /usr/

HEALTHCHECK NONE
ENTRYPOINT []
CMD ["/bin/bash"]
