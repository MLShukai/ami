FROM ubuntu:22.04

# Setup workspace
RUN mkdir /workspace
WORKDIR /workspace
ADD ./ /workspace/

# Install dependencies
ARG DEBIAN_FRONTEND=noninteractive
RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime
# Change apt source to japan
# For linux host (x86_64)
RUN sed -i -e 's|archive.ubuntu.com/ubuntu|ftp.naist.jp/pub/Linux/ubuntu|g' /etc/apt/sources.list
# For linux or mac host (arm64)
RUN sed -i -e 's|ports.ubuntu.com|jp.mirror.coganng.com|g' /etc/apt/sources.list

RUN apt-get update && apt-get install -y \
    curl \
    git \
    make \
    tmux \
    python3 \
    python3-pip \
    libopencv-dev \
    xvfb \
    x11-utils \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libhdf5-dev \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
RUN python3.10 -m pip install poetry && \
    poetry install && \
    echo "cd /workspace && poetry shell" >> ~/.bashrc && \
    echo "eval '$(poetry run python scripts/launch.py -sc install=bash)'" >> ~/.bashrc

# Copy Xvfb script
COPY scripts/start-xvfb.sh /start-xvfb.sh
RUN chmod +x /start-xvfb.sh


# Set DISPLAY environment variable
ENV DISPLAY=:99

# Set entrypoint to run Xvfb
ENTRYPOINT ["/start-xvfb.sh"]

# Default command (can be overridden)
CMD ["/bin/bash"]
