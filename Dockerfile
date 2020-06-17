FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt -qq update && apt -qq install -y --no-install-recommends \
    wget \
    ca-certificates \
    locales \
    libglib2.0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    xvfb \
    ffmpeg \
    freeglut3-dev \
 && rm -rf /var/cache/*

# S3 client
RUN wget -nv -O ./mc https://dl.min.io/client/mc/release/linux-amd64/mc \
 && mv ./mc /bin/mc && chmod +x /bin/mc

# Unicode support:
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Create user home directory
ENV USER_NAME aicrowd
ENV HOME_DIR /home/$USER_NAME

# Replace HOST_UID/HOST_GUID with your user / group id
ENV HOST_UID 1001
ENV HOST_GID 1001

# Use bash as default shell, rather than sh
ENV SHELL /bin/bash

# Set up user
RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${HOST_UID} \
    ${USER_NAME}

COPY . ${HOME_DIR}

RUN chown -R ${USER_NAME}:${USER_NAME} ${HOME_DIR}

USER ${USER_NAME}
WORKDIR ${HOME_DIR}

ENV CONDA_DIR ${HOME_DIR}/.conda

RUN wget -nv -O miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.2-Linux-x86_64.sh \
 && bash miniconda.sh -b -p ${CONDA_DIR} \
 && . ${CONDA_DIR}/etc/profile.d/conda.sh \
 && rm -rf miniconda.sh

ENV PATH ${CONDA_DIR}/bin:${PATH}

RUN pip install -r requirements.txt --no-cache-dir
