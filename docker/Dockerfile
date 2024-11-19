# FROM nvcr.io/nvidia/pytorch:23.07-py3
FROM nvcr.io/nvidia/pytorch:23.04-py3
# FROM nvcr.io/nvidia/pytorch:22.12-py3
# FROM nvcr.io/nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

## * ==================== Install SAPEON SDK Dependencies ================= * ##
ARG PYTHON_VERSION=python3.8
RUN apt-get update && apt-get -y install software-properties-common && \
    add-apt-repository -y ppa:ubuntu-toolchain-r/test && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get -y install \
    gcc g++ \
    cmake curl wget \
    gdb git vim  \
    libx11-6 \
    ${PYTHON_VERSION} ${PYTHON_VERSION}-dev ${PYTHON_VERSION}-distutils \
    libgoogle-glog-dev gcc g++ libzip-dev libstdc++6 \
    libboost-dev libboost-serialization-dev libgtest-dev \
    libeigen3-dev \
    libgl1-mesa-glx && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    ${PYTHON_VERSION} get-pip.py && \
    ${PYTHON_VERSION} -m pip install --upgrade pip setuptools wheel pybind11 pybind11-global numpy && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# To solve issue related to 
# "libcrypto.so.1.1: cannot open shared object file: No such file or directory" 
# in Ubuntu 20.04, 22.04, install openssl 1.1.1q
# It will fix the build issue of sapeon_compiler.
RUN mkdir -p /opt/openssl && \
    wget https://www.openssl.org/source/openssl-1.1.1q.tar.gz --no-check-certificate -P /opt/openssl/ && \
    tar -xzvf /opt/openssl/openssl-1.1.1q.tar.gz -C /opt/openssl/  && \
    cd /opt/openssl/openssl-1.1.1q && \
    ./config  && \
    make install && \
    cp /usr/local/bin/openssl /usr/bin/openssl

ARG PB_VERSION="3.16.0"
RUN wget https://github.com/protocolbuffers/protobuf/releases/download/v${PB_VERSION}/protobuf-all-$PB_VERSION.tar.gz -P /tmp --no-check-certificate  && \
    cd /tmp && \
    tar xvzf protobuf-all-${PB_VERSION}.tar.gz && \
    cd /tmp/protobuf-${PB_VERSION} && \
    ./configure && \
    make -j $(nproc) && \
    make install && \
    ldconfig

RUN cd /usr/lib/x86_64-linux-gnu/ && \
    ln -s libzip.so.5 libzip.so.4

RUN wget https://github.com/google/glog/archive/refs/tags/v0.6.0.tar.gz -P /tmp --no-check-certificate && \
    cd /tmp && \
    tar xvzf v0.6.0.tar.gz && \
    cd /tmp/glog-0.6.0 && \
    mkdir -p build && \
    cd build && \
    cmake ..  && \
    make -j `nproc` && \
    make install

## * ==================== Install SAPEON SDK Dependencies ================= * ##

## * ==================== Install MMDetection3D Dependencies ============== * ##


RUN pip install -U \
    openmim \
    opencv-python==4.8.0.74 \
    numpy==1.23.5 numba \
    onnxsim onnxoptimizer

# Install MMEngine, MMCV and MMDetection
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    FORCE_CUDA="1"

RUN mim install  \
    "mmengine==0.10.3" \
    "mmcv-full==1.6.0" \ 
    "mmcls==0.25.0" \
    "mmdet==2.28.2" \
    "mmsegmentation==0.30.0"


## * ==================== Install MMDetection3D Dependencies ============== * ##
