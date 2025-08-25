# hash:sha256:e70a5cf1e49dd28853e53079c59d19d1decbfe6acf15b7102902e44a2b66a349
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/pytorch-tensorflow-jax:2.1

ARG DEBIAN_FRONTEND=noninteractive

RUN conda create -n cell_class python=3.9

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "cell_class", "/bin/bash", "-c"]

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN apt-get update
RUN apt-get -y install gcc g++ libcublas-12-0 libpcre3

RUN conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

RUN pip install -U --no-cache-dir \
    git+https://github.com/AllenNeuralDynamics/aind-large-scale-prediction.git@unpinned-versions \
    aind-data-schema==1.0.0 \
    aind-ng-link==1.0.6 \
    psutil==5.9.8 \
    jupyter-server-proxy==3.2.2 \
    cellfinder>=1.0.0 \
    argschema \
    pandas \
    openpyxl \
    keras==3.10.0 \
    tensorflow==2.18.0
