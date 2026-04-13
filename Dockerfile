FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y \
    python3 \
    python3-pip \
    libpcl-dev \
    pkg-config \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 👉 Critical fix: set PKG_CONFIG_PATH
ENV PKG_CONFIG_PATH=/usr/lib/x86_64-linux-gnu/pkgconfig

# Upgrade pip (important for old packages)
RUN pip3 install --upgrade pip setuptools wheel

# ✅ Install Cython FIRST
RUN pip3 install cython
RUN pip3 install numpy

# # RUN pkg-config --list-all | grep pcl
# RUN cd /usr/lib/x86_64-linux-gnu/pkgconfig && \
#     for f in pcl_*.pc; do \
#         cp "$f" "$(basename $f .pc)-1.12.pc"; \
#     done
# # Install python-pcl
# RUN pip3 install python-pcl --no-build-isolation

WORKDIR /app