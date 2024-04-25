#!/bin/bash

# Specify the Miniconda installer script
MINICONDA_INSTALLER_SCRIPT=Miniconda3-latest-Linux-x86_64.sh
MINICONDA_INSTALL_PATH="$HOME/miniconda3"

# Download Miniconda installer
wget https://repo.anaconda.com/miniconda/$MINICONDA_INSTALLER_SCRIPT

# Make the installer script executable
chmod +x $MINICONDA_INSTALLER_SCRIPT

# Install Miniconda without user interaction
./$MINICONDA_INSTALLER_SCRIPT -b -p $MINICONDA_INSTALL_PATH

# Initialize Conda for all shells (might need to adjust this depending on shell preference)
$MINICONDA_INSTALL_PATH/bin/conda init

# Wait a moment to ensure the initialization takes effect
sleep 1

# Clean up the installer script
rm $MINICONDA_INSTALLER_SCRIPT

# Add Conda init to the current shell session
if [ -f "$HOME/.bashrc" ]; then
    source $HOME/.bashrc
elif [ -f "$HOME/.bash_profile" ]; then
    source $HOME/.bash_profile
fi

# Create a new conda environment named 'hf'
$MINICONDA_INSTALL_PATH/bin/conda create --name hf -y

# Directly activate the conda environment using its path
source $MINICONDA_INSTALL_PATH/bin/activate hf
exit

conda activate hf
pip install datasets transformers bitsandbytes num2words word2number spacy
python -m spacy download en_core_web_sm
