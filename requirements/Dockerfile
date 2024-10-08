# Arguments to pass to the image
ARG TF_VER=tf2
ARG VERSION_DATE=24.08
ARG FROM_IMAGE=nvcr.io/nvidia/tensorflow

# Import RAPIDS container as the BASE Image (cuda base image)
FROM ${FROM_IMAGE}:${VERSION_DATE}-${TF_VER}-py3

# Ubuntu needs noninteractive to be forced
ENV DEBIAN_FRONTEND noninteractive
ENV CPLUS_INCLUDE_PATH="/usr/include/gdal"
ENV C_INCLUDE_PATH="/usr/include/gdal"
ENV PROJ_LIB="/usr/share/proj"
ENV SM_FRAMEWORK="tf.keras"

# System dependencies
RUN apt-get update && \
    apt-get -y install software-properties-common && \
    add-apt-repository ppa:ubuntugis/ubuntugis-unstable && \
    apt-get update && \
    apt-get -y dist-upgrade && \
    apt-get -y install \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libfontconfig1 \
        bzip2 \
        diffutils \
        file \
        build-essential \
        make \
        swig \
        libnetcdf-dev \
        libacl1-dev \
        libgeos++-dev \
        libgeos-dev \
        libsqlite3-dev \
        libx11-dev \
        libspatialindex-dev \
        wget \
        vim \
        curl \
        git \
        procps \
        gcc \
        g++ \
        bzip2 \
        libssl-dev \
        libzmq3-dev \
        libpng-dev \
        libfreetype6-dev \ 
        locales \
        gdal-bin \
        libgdal-dev && \
    apt-get -y autoremove && \
    rm -rf /var/cache/apt /var/lib/apt/lists/*


# Install shiftc
WORKDIR /app
RUN git clone --single-branch --branch master https://github.com/pkolano/shift.git && \
    cd shift/c && \
    make nolustre && \
    cd ../ && \
    install -m 755 perl/shiftc /usr/local/bin/ && \
    install -m 755 c/shift-bin /usr/local/bin/ && \
    install -m 755 perl/shift-mgr /usr/local/bin/ && \
    install -m 644 etc/shiftrc /etc/ && \
    install -m 755 perl/shift-aux /usr/local/bin/ && \
    install -m 755 c/shift-bin /usr/local/bin/ && \
    export LC_ALL=en_US.UTF-8 && \
    export LANG=en_US.UTF-8 && \
    locale-gen en_US.UTF-8 && \
    rm -rf /app

# Pip
RUN pip install \
    scipy \
    omegaconf \
    rasterio \
    rioxarray \
    xarray \
    geopandas \
    tensorflow_probability \
    keras-cv \
    keras-unet-collection \
    opencv-python \
    opencv-python-headless \
    opencv-contrib-python \
    opencv-contrib-python-headless \
    tifffile \
    webcolors \
    Pillow \
    seaborn \
    xgboost \
    tiler \
    segmentation-models \
    pytest \
    coveralls \
    rtree \
    rasterstats \
    sphinx \
    sphinx_rtd_theme \
    geedim \
    geemap \
    GDAL==`ogrinfo --version | grep -Eo '[0-9]\.[0-9]\.[0-9]+'` \
    zarr \
    cubo[ee] \
    git+https://github.com/dshean/pygeotools.git@master \
    git+https://github.com/nasa-nccs-hpda/tensorflow-caney.git@main

HEALTHCHECK NONE
ENTRYPOINT []
CMD ["/bin/bash"]
