# hash:sha256:7bcd136cdf875a55ea94f58662c2d4e4b7aae6db20d08f16893a9bf0703dc4d3
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/tensorflow:2.11-python3.10-cuda11.7.0-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN conda create -n cell_seg python=3.8

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "cell_seg", "/bin/bash", "-c"]

RUN apt-get update
RUN apt-get -y install gcc g++ libcublas-12-0 libpcre3

RUN pip install -U --no-cache-dir \
    git+https://github.com/camilolaiton/cellfinder-core.git@block_seg \
    bokeh==2.4.2 \
    dask[distributed]==2022.11.1 \
    dask_memusage \
    argschema==3.0.4 \
    natsort==8.2.0 \
    astropy==5.2.1 \
    photutils==1.6.0 \
    imlib==0.1.3 \
    zarr==2.13.6 \
    aind-data-schema==1.0.0 \
    aind-ng-link==1.0.6 \
    psutil==5.9.8 \
    jupyter-server-proxy==3.2.2 \
    awscli