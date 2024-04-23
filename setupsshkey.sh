#!/bin/bash

# Configuration
SSH_DIR="$HOME/.ssh"
SSH_KEY_NAME="runpod"
SSH_PRIVATE_KEY_PATH="$SSH_DIR/$SSH_KEY_NAME"

# Create the SSH directory if it doesn't exist
mkdir -p "$SSH_DIR"
chmod 700 "$SSH_DIR"

# Prompt the user to paste their SSH private key
echo "Please paste your SSH private key for the runpod key, then press Ctrl-D:"
cat > "$SSH_PRIVATE_KEY_PATH"

# Change the permissions of the SSH private key to be read-only by the user
chmod 600 "$SSH_PRIVATE_KEY_PATH"

# Start the ssh-agent and add the key
eval "$(ssh-agent -s)"
ssh-add "$SSH_PRIVATE_KEY_PATH"

echo "SSH key setup complete."
