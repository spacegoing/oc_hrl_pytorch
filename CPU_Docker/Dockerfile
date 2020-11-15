FROM ubuntu:18.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
  libglib2.0-0 libxext6 libsm6 libxrender1 \
  git mercurial subversion

RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh -O ~/anaconda.sh && \
  /bin/bash ~/anaconda.sh -b -p /opt/conda && \
  rm ~/anaconda.sh && \
  ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
  echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
  echo "conda activate base" >> ~/.bashrc

RUN apt-get install -y curl grep sed dpkg && \
  TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
  curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
  dpkg -i tini.deb && \
  rm tini.deb && \
  apt-get clean

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated --no-install-recommends \
  build-essential apt-utils cmake git curl vim ca-certificates \
  libjpeg-dev libpng-dev libgtk3.0 libsm6 \
  qtbase5-dev libqt5opengl5-dev libassimp-dev \
  libtinyxml-dev bash \
  wget unzip libopenmpi-dev software-properties-common \
  software-properties-common net-tools xpra xserver-xorg-dev \
  libosmesa6-dev libglu1-mesa-dev libgl1-mesa-dev libgl1-mesa-glx libglew-dev libglfw3-dev \
  xvfb patchelf swig libglfw3 libglew2.0 python3-dev zlib1g-dev

RUN conda install -y pytorch torchvision torchaudio cpuonly -c pytorch
RUN conda install -y tensorflow
RUN git clone https://github.com/spacegoing/sa_docker_dependencies.git
RUN mv /sa_docker_dependencies/.mujoco /root/.mujoco

ENV LD_LIBRARY_PATH /usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}
COPY ./mjkey.txt /root/.mujoco/mjkey.txt
COPY ./mjkey.txt /root/.mujoco/mjpro150/mjkey.txt
COPY ./mjkey.txt /root/.mujoco/mjpro150/bin/mjkey.txt
COPY ./mjkey.txt /root/.mujoco/mjpro200_linux/mjkey.txt
COPY ./mjkey.txt /root/.mujoco/mjpro200_linux/bin/mjkey.txt
RUN pip install /sa_docker_dependencies/mujoco-py/
RUN pip install -e /sa_docker_dependencies/gym/[all]

WORKDIR /root/
EXPOSE 27017
EXPOSE 6006
