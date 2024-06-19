#!/bin/bash

# Define the path to the log wrapper script
LOG_WRAPPER_PATH="/workspace/slice-monorepo/log_wrapper.py"

# Define the target directory for the bash wrapper script
TARGET_DIR="/usr/local/bin"

# Create the bash wrapper script
echo "#!/bin/bash" > $TARGET_DIR/log
echo "python $LOG_WRAPPER_PATH \"\$@\"" >> $TARGET_DIR/log

# Ensure the bash wrapper script is executable
chmod +x $TARGET_DIR/log

# Add /usr/local/bin to PATH if it's not already in PATH
if [[ ":$PATH:" != *":$TARGET_DIR:"* ]]; then
  echo "export PATH=\$PATH:$TARGET_DIR" >> ~/.bashrc
  source ~/.bashrc
fi

echo "Setup complete. You can now use the 'log' command from anywhere."


mkdir "/workspace/logs"