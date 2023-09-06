#!/bin/bash

# Install required packages
pip3 install -U scipy
pip3 install transformers==4.19.0

# Clone the Tortoise TTS repository
git clone https://github.com/jnordberg/tortoise-tts.git

# Navigate to the cloned directory
cd tortoise-tts

# Install other dependencies
pip3 install -r requirements.txt

# Install Tortoise TTS
python3 setup.py install
