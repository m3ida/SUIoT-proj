# Use the specified base image
FROM ultralytics/ultralytics:latest-jetson-jetpack4

# Set the working directory
WORKDIR /proj

# Create a user with UID 1000 and GID 1000
RUN groupadd -g 1000 myuser && \
    useradd -m -u 1000 -g myuser myuser

RUN apt-get update &&\
    DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata apt-utils

# Install GStreamer and NVIDIA plugins
RUN apt-get update && \
apt-get -y upgrade && \
    apt-get install -y \
    python3-pip \
    gir1.2-gstreamer-1.0 \
    gstreamer1.0-alsa \
    gstreamer1.0-clutter-3.0 \
    gstreamer1.0-gl \
    gstreamer1.0-gtk3 \
    gstreamer1.0-libav \
    gstreamer1.0-nice \
    gstreamer1.0-packagekit \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-base-apps \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-pulseaudio \
    gstreamer1.0-tools \
    gstreamer1.0-x \
    libgstreamer-gl1.0-0 \
    libgstreamer-opencv1.0-0 \
    libgstreamer-plugins-bad1.0-0 \
    libgstreamer-plugins-bad1.0-dev \
    libgstreamer-plugins-base1.0-0 \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-good1.0-0 \
    libgstreamer-plugins-good1.0-dev \
    libgstreamer1.0-0 \
    libgstreamer1.0-dev \
    build-essential \
    cmake \
    git \
    pkg-config \
    libgtk-3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    gfortran \
    && pip3 install setuptools==68 && \
    pip3 install Jetson.GPIO

WORKDIR /opt
RUN git clone --depth 1 https://github.com/opencv/opencv.git && \
    git clone --depth 1 https://github.com/opencv/opencv_contrib.git 

WORKDIR /opt/opencv/build
RUN cmake -D CMAKE_BUILD_TYPE=Release \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D ENABLE_PRECOMPILED_HEADERS=OFF \
          -D WITH_GSTREAMER=ON \
          -D WITH_V4L=ON \
          -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib/modules \
          -D BUILD_EXAMPLES=OFF \
          -D BUILD_TESTS=OFF \ 
          -D WITH_OPENMP=OFF \ 
          -D BUILD_PERF_TESTS=OFF \
          -D BUILD_opencv_python3=ON \
          -D OPENCV_GENERATE_PKGCONFIG=ON \
          -D PYTHON3_EXECUTABLE=/usr/bin/python3.8 \
          -D PYTHON3_INCLUDE_DIR=/usr/include/python3.8 \
          -D PYTHON3_NUMPY_INCLUDE_DIRS=/usr/local/lib/python3.8/dist-packages/numpy/core/include \
            ..
RUN make -j$(nproc) && make install && ldconfig
RUN python3 -c "import cv2; print(cv2.__version__); print(cv2.getBuildInformation())"
RUN export PYTHONPATH=/opt/opencv/build/lib/python3/:$PYTHONPATH
# Switch to the new user
USER myuser

# Set the entry point to keep the container running interactively
CMD ["bash"]
