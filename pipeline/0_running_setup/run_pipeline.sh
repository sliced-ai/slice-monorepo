#!/bin/bash

# Activate the conda environment
source activate finetune

# Start the service in the background and save its process ID (PID)
python app.py &
SERVICE_PID=$!

# Execute the first command
python demographics.py

# After the command finishes, kill the service process
kill $SERVICE_PID

echo "Pipeline completed and service stopped."
