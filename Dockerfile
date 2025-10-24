# This is an auto generated Dockerfile for ros:perception
# generated from docker_images/create_ros_image.Dockerfile.em
FROM ubuntu:22.04

# Maintainer instructions has been deprecated, instead use LABEL
LABEL maintainer="tannousgeagea@hotmail.com"

# Versionining as "b-beta, a-alpha, rc - release candidate"
LABEL com.wasteant.version="1.1b1"

# [CHECK] Whether it is convenient to use the local user values or create ENV variables, or run everyhting with root
ARG user=appuser
ARG userid=1000
ARG group=appuser
ARG groupid=1000
ARG installation_folder="./installation_files"

# Install other necessary packages and dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -q -y --no-install-recommends \
    apt-utils \
	vim \
	git \
	iputils-ping \
	net-tools \
	netcat \
	ssh \
    curl \
    lsb-release \
    wget \
    zip \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies to build your own ROS packages
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -q -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3 \
    python3-pip \
	python3-wstool\
    build-essential \
	python3-pip \
	python3-distutils \
	python3-psutil \
    python3-tk \
    git \
	ffmpeg \
	&& rm -rf /var/lib/apt/lists/*

# Install libraries
RUN pip3 install opencv-python
RUN pip3 install natsort
RUN pip3 install schedule
RUN pip3 install numpy
RUN pip3 install pandas
RUN pip3 install matplotlib
RUN pip3 install pymongo
RUN pip3 install imutils
RUN pip3 install fpdf
RUN pip3 install pytz
RUN pip3 install pyyaml
RUN pip3 install tqdm
RUN pip3 install supervisor
RUN pip3 install fastapi[standard]
RUN pip3 install uvicorn[standard]
RUN pip3 install flower
RUN pip3 install redis
RUN pip3 install celery
RUN pip3 install asgi_correlation_id
RUN pip3 install django==4.2
RUN pip3 install gunicorn
RUN pip3 install requests
RUN pip3 install python-redis-lock
RUN pip3 install grpcio
RUN pip3 install grpcio-tools
RUN pip3 install confluent-kafka
RUN pip3 install scp
RUN pip3 install django-unfold
RUN pip3 install django-storages[azure]
RUN pip3 install psycopg2-binary
RUN pip3 install python-jose
RUN pip3 install zipstream
RUN pip3 install albumentations
RUN pip3 install perceptra-storage[all]
RUN pip3 install pyjwt

# upgrade everything
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get upgrade -q -y \
   && rm -rf /var/lib/apt/lists/*

# # Set up users and groups
RUN addgroup --gid $groupid $group && \
	adduser --uid $userid --gid $groupid --disabled-password --gecos '' --shell /bin/bash $user && \
	echo "$user ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$user && \
	chmod 0440 /etc/sudoers.d/$user

# # # Create initial workspace 
RUN mkdir -p /home/$user/src

RUN /bin/bash -c "chown -R $user:$user /home/$user/"
RUN /bin/bash -c "chown -R $user:$user /media"

# Create directory for Supervisor logs
RUN mkdir -p /var/log/supervisor && \
    chmod -R 755 /var/log/supervisor
	

COPY . /home/${user}/src
COPY ./supervisord.conf /etc/supervisord.conf
COPY ./entrypoint.sh /home/.
RUN /bin/bash -c "chown -R $user:$user /home/$user/"
RUN /bin/bash -c "chown $user:$user /home/entrypoint.sh"

ENTRYPOINT /bin/bash -c ". /home/entrypoint.sh"
