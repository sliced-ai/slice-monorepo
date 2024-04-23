#!/bin/bash

# Specify the Miniconda installer script
MINICONDA_INSTALLER_SCRIPT=Miniconda3-latest-Linux-x86_64.sh

# Download Miniconda installer
wget https://repo.anaconda.com/miniconda/$MINICONDA_INSTALLER_SCRIPT

# Make the installer script executable
chmod +x $MINICONDA_INSTALLER_SCRIPT

# Install Miniconda without user interaction
./$MINICONDA_INSTALLER_SCRIPT -b

# Initialize Conda
~/miniconda3/bin/conda init

# Wait a moment to ensure the initialization takes effect
sleep 1

# Activate the conda command
source ~/.bashrc

# Create a new conda environment named 'hf'
conda create --name hf -y

# Activate the new environment
conda activate hf

# Clean up the installer script
rm $MINICONDA_INSTALLER_SCRIPT

echo "The 'hf' conda environment is ready and activated."
