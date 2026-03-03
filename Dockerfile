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
    python3-dev \
    python3-pip \
    python3-venv \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Set working directory
WORKDIR /workspace

# Clone diffusion-pipe with submodules
RUN git clone --recurse-submodules https://github.com/tdrussell/diffusion-pipe.git
WORKDIR /workspace/diffusion-pipe

# Install PyTorch 2.10 + torchvision compatible with CUDA 12.8
RUN python3 -m pip install --no-cache-dir \
    torch==2.10.0+cu128 \
    torchvision==0.25.0+cu128 \
    torchaudio==2.10.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128
	
# Install DeepSpeed with CUDA ops
RUN DS_BUILD_OPS=1 python3 -m pip install --no-cache-dir deepspeed

# Install remaining Python dependencies
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Install Jupyter
RUN python3 -m pip install --no-cache-dir notebook

# Jupyter environment variables
ENV JUPYTER_PORT=8888
ENV JUPYTER_TOKEN=z

EXPOSE 8888

# Start Jupyter Notebook
CMD jupyter notebook \
    --NotebookApp.token=${JUPYTER_TOKEN} \
    --NotebookApp.ip=0.0.0.0 \
    --port=${JUPYTER_PORT} \
    --allow-root
