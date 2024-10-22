#!/bin/bash

# Update package list and ensure pip is installed
echo "Updating package list and installing pip if necessary..."
apt update

# Upgrade pip to the latest version
echo "Upgrading pip to the latest version..."
pip3 install --upgrade pip

# Install required Python packages
echo "Installing transformers, pandas, and matplotlib..."
pip3 install transformers pandas matplotlib datasets accelerate seaborn scikit-learn tensorboard

# Confirm the installation
echo "Installed packages:"
pip3 show transformers pandas matplotlib datasets accelerate seaborn scikit-learn tensorboard

echo "Installation completed successfully!"
