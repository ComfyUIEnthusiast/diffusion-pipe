FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# Non-interactive frontend
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="8.9" 

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    curl \
    ninja-build \
	cmake \
    && rm -rf /var/lib/apt/lists/*
	
RUN apt-get update && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y \
        python3.12 \
        python3.12-venv \
        libpython3.12-dev \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.12
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Make python3 point to 3.12
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Set working directory
WORKDIR /workspace

# Clone diffusion-pipe with submodules
RUN git clone --recurse-submodules https://github.com/tdrussell/diffusion-pipe.git
WORKDIR /workspace/diffusion-pipe

# Upgrade build tools
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch 2.10 + torchvision compatible with CUDA 12.8
RUN python3 -m pip install --no-cache-dir \
    torch==2.9.0+cu128 \
    torchvision==0.24.0+cu128 \
    torchaudio==2.9.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

#Install flash-attn
RUN python3 -m pip install --no-cache-dir \
"https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

# Install DeepSpeed with CUDA ops
RUN python3 -m pip install --no-cache-dir deepspeed --no-build-isolation

# Install remaining Python dependencies
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Install Jupyter
RUN python3 -m pip install --no-cache-dir notebook

# Jupyter environment variables
ENV JUPYTER_PORT=8888
ENV JUPYTER_TOKEN=z

EXPOSE 8888

WORKDIR /

# Start Jupyter Notebook
CMD jupyter notebook \
    --NotebookApp.token=${JUPYTER_TOKEN} \
    --NotebookApp.ip=0.0.0.0 \
    --port=${JUPYTER_PORT} \
    --allow-root
