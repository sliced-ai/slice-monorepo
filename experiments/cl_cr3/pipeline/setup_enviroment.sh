#!/bin/bash

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# 1. Check if conda is installed
if ! command_exists conda; then
    echo "Conda is not installed. Please install Conda and try again."
    exit 1
fi

echo "Conda is installed."

# 2. Create a conda environment
ENV_NAME="new_rapids_env"
PYTHON_VERSION="3.11"
RAPIDS_VERSION="24.06"
CUDA_VERSION="12.2"

echo "Creating conda environment '$ENV_NAME' with RAPIDS $RAPIDS_VERSION, Python $PYTHON_VERSION, and CUDA $CUDA_VERSION..."
conda create -n $ENV_NAME -c rapidsai -c conda-forge -c nvidia rapids=$RAPIDS_VERSION python=$PYTHON_VERSION cuda-version=$CUDA_VERSION -y

# Activate the environment
source activate $ENV_NAME

# 3. Install requirements file
REQUIREMENTS_FILE="requirements.txt"

if [ ! -f $REQUIREMENTS_FILE ]; then
    echo "Requirements file not found: $REQUIREMENTS_FILE"
    exit 1
fi

echo "Installing packages from $REQUIREMENTS_FILE..."
pip install -r $REQUIREMENTS_FILE

# 4. Any other installation needed
echo "Installing additional packages..."
# Add any additional installation commands here

echo "Environment setup is complete. Activate the environment using 'conda activate $ENV_NAME'."
