# hash:sha256:7bcd136cdf875a55ea94f58662c2d4e4b7aae6db20d08f16893a9bf0703dc4d3
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/tensorflow:2.11-python3.10-cuda11.7.0-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN apt-get update
RUN apt-get -y install gcc g++ libcublas-12-0 libpcre3

RUN pip install -U --no-cache-dir \
    aind-large-scale-prediction==1.0.0 \
    aind-data-schema==1.0.0 \
    aind-ng-link==1.0.6 \
    psutil==5.9.8 \
    jupyter-server-proxy==3.2.2 \
    cellfinder>=1.0.0 \
    argschema
