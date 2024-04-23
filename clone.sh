#!/bin/bash

# Configuration
SSH_DIR="$HOME/.ssh"
SSH_KEY_NAME="runpod"
SSH_PRIVATE_KEY_PATH="$SSH_DIR/$SSH_KEY_NAME"
REPO_URL="git@github.com:sliced-ai/slice-monorepo.git"
BRANCH="runpod"

# Ensure the SSH directory exists with the correct permissions
mkdir -p "$SSH_DIR"
chmod 700 "$SSH_DIR"

# Prompt the user to paste their private SSH key
echo "Please paste your SSH private key for the runpod key:"
cat > "$SSH_PRIVATE_KEY_PATH"
chmod 600 "$SSH_PRIVATE_KEY_PATH"

# Add a newline to the end of the key to ensure correct EOF
echo >> "$SSH_PRIVATE_KEY_PATH"

# Start the ssh-agent and add the key
eval "$(ssh-agent -s)"
ssh-add "$SSH_PRIVATE_KEY_PATH"

# Now clone the repository
git clone -b "$BRANCH" "$REPO_URL"

echo "Repository cloned successfully."
