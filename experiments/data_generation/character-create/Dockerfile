# Use an official NVIDIA CUDA base image as a parent image
FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04

# Set the working directory in Docker
WORKDIR /tmp
RUN mkdir ./data
# Copy the current directory contents into the container at /tmp
COPY ./ ./

# Install Miniconda (replace with the appropriate Miniconda installer URL)
RUN apt-get update && apt-get install -y curl && \
    curl -o Miniconda3-latest-Linux-x86_64.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x Miniconda3-latest-Linux-x86_64.sh && \
    ./Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    apt-get remove -y curl && apt-get clean && apt-get autoclean && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# Update Conda and install the environment
ENV PATH=/opt/conda/bin:$PATH
ENV CONDA_ENV=finetune
RUN conda update -n base conda && \
    conda env create -f environment.yml
    
ENV PYTHONUNBUFFERED=1

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "$CONDA_ENV", "/bin/bash", "-c"]
