# Arguments to pass to the image
ARG TF_VER=tf2
ARG VERSION_DATE=22.08
ARG FROM_IMAGE=nvcr.io/nvidia/tensorflow

#FROM osgeo/gdal:ubuntu-full-latest as builder

# Import RAPIDS container as the BASE Image (cuda base image)
FROM ${FROM_IMAGE}:${VERSION_DATE}-${TF_VER}-py3

# Ubuntu needs noninteractive to be forced
ENV DEBIAN_FRONTEND noninteractive
ENV LD_LIBRARY_PATH="/mnt/data-store/FileGDB_API-64gcc51/lib/:/usr/local/lib"
ENV PROJ_LIB="/usr/share/proj"
ENV CPLUS_INCLUDE_PATH="/usr/include/gdal"
ENV C_INCLUDE_PATH="/usr/include/gdal"

# global definitions for gdal
#ARG dataStoreDir="/mnt/data-store/"
#ARG gdalVersion=3.2.0
#ARG gdalName=gdal-$gdalVersion

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

# install geos, FileGDB, and gdal
#RUN mkdir -p $dataStoreDir && \
#    cd $dataStoreDir && \
#    wget http://download.osgeo.org/geos/geos-3.8.1.tar.bz2 && \
#    tar -xjf geos-3.8.1.tar.bz2 && \
#    rm geos-3.8.1.tar.bz2 && \
#    cd $dataStoreDir\geos-3.8.1 && \
#    ./configure --enable-python --enable-static --with-pic && \
#    make -j 10 && make install && \
#    cd $dataStoreDir && \
#    wget https://github.com/Esri/file-geodatabase-api/raw/master/FileGDB_API_1.5/FileGDB_API_1_5_64gcc51.tar.gz && \
#    tar -zxf FileGDB_API_1_5_64gcc51.tar.gz && rm FileGDB_API_1_5_64gcc51.tar.gz && \
#    cd $dataStoreDir && wget http://download.osgeo.org/gdal/${gdalVersion}/${gdalName}.tar.gz && \
#    tar -zxf ${gdalName}.tar.gz && rm ${gdalName}.tar.gz && \
#	cd $dataStoreDir/$gdalName && \
#    ./configure LIBTOOL='/usr/bin/libtool --tag=CC' --with-python --enable-static --with-pic -with-geos=yes \
#        --with-netcdf --with-fgdb=/mnt/data-store/FileGDB_API-64gcc51 && \
#	export LD_LIBRARY_PATH=/mnt/data-store/FileGDB_API-64gcc51/lib/:/usr/local/lib && \
#	make -j 10 && make install

# Pip
RUN pip --no-cache-dir install omegaconf rasterio rioxarray xarray geopandas \
    opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless \
    tifffile webcolors Pillow seaborn xgboost segmentation-models pytest coveralls \
    GDAL==`ogrinfo --version | grep -Eo '[0-9]\.[0-9]\.[0-9]+'`

#ARG PROJ_INSTALL_PREFIX=/usr/local
#COPY --from=builder  ${PROJ_INSTALL_PREFIX}/share/proj ${PROJ_INSTALL_PREFIX}/share/proj
#COPY --from=builder  ${PROJ_INSTALL_PREFIX}/include/* ${PROJ_INSTALL_PREFIX}/include
#COPY --from=builder  ${PROJ_INSTALL_PREFIX}/bin/ ${PROJ_INSTALL_PREFIX}/bin/
#COPY --from=builder  ${PROJ_INSTALL_PREFIX}/lib/lib* ${PROJ_INSTALL_PREFIX}/lib
#COPY --from=builder  /usr/share/java /usr/share/java
#COPY --from=builder  /usr/share/gdal/ /usr/share/gdal/
#COPY --from=builder  /usr/include/ /usr/include/
#COPY --from=builder  /usr/ /usr/

HEALTHCHECK NONE
ENTRYPOINT []
CMD ["/bin/bash"]
