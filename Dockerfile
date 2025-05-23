# syntax=docker/dockerfile:1
FROM continuumio/miniconda3:latest
#Set the timezone so installs don't ask for it
RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone

#install python, pip, and git
RUN apt-get update
RUN apt-get full-upgrade -y
RUN apt-get install -y git 
RUN apt-get install -y git-lfs wget build-essential

#clone ComfyUI
RUN git clone --recurse-submodules https://github.com/tdrussell/diffusion-pipe

#makes future installations faster


RUN conda install -y cuda -c nvidia
RUN conda create -n diffusion-pipe python=3.12
ENV PATH /opt/conda/envs/diffusion-pipe/bin:$PATH
ADD requirements.txt .
RUN pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
#RUN pip install -r requirements.txt
WORKDIR /diffusion-pipe
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
#RUN pip install -r requirements.txt

ENV JUPYTER_PORT=$JUPYTER_PORT
ENV JUPYTER_TOKEN=$JUPYTER_TOKEN
WORKDIR /configs
ADD dataset.toml .
ADD config.toml .
WORKDIR /
ADD start.sh ./
CMD ["sh", "./start.sh"]