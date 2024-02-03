FROM ubuntu:22.04

# Setup workspace
RUN mkdir /workspace
WORKDIR /workspace
ADD ./ /workspace/

# Install dependencies
ARG DEBIAN_FRONTEND=noninteractive
RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime
RUN perl -p -i.bak -e 's%(deb(?:-src|)\s+)https?://(?!archive\.canonical\.com|security\.ubuntu\.com)[^\s]+%$1http://ftp.naist.jp/pub/Linux/ubuntu/%' /etc/apt/sources.list
RUN apt-get update && apt-get install -y \
    curl \
    git \
    make \
    screen \
    software-properties-common \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

RUN add-apt-repository ppa:deadsnakes/ppa -y && apt-get update && apt-get install -y \
    python3.11 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
RUN python3.11 -m pip install poetry && \
    poetry install && \
    echo "cd /workspace && poetry shell" >> ~/.bashrc
