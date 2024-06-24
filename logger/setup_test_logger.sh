#!/bin/bash

# Define paths
LOG_DIR="/workspace/logs"
LOG_WRAPPER_PATH="/workspace/slice-monorepo/logger/log_wrapper.py"
TARGET_DIR="/usr/local/bin"
TEST_DIR="/workspace/slice-monorepo/logger/test_dir"
SAVE_LOCATION="$TEST_DIR/save_location"

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$TARGET_DIR"
mkdir -p "$TEST_DIR"
mkdir -p "$SAVE_LOCATION/logs"

# Create the bash wrapper script
echo "#!/bin/bash" > "$TARGET_DIR/log"
echo "python $LOG_WRAPPER_PATH \"\$@\"" >> "$TARGET_DIR/log"
chmod +x "$TARGET_DIR/log"

# Add /usr/local/bin to PATH if it's not already in PATH
if [[ ":$PATH:" != *":$TARGET_DIR:"* ]]; then
  echo "export PATH=\$PATH:$TARGET_DIR" >> ~/.bashrc
  source ~/.bashrc
fi

# Create a test Python script
PYTHON_SCRIPT="$LOG_DIR/test_script.py"
cat <<EOF > "$PYTHON_SCRIPT"
import sys
import os

def main():
    if len(sys.argv) < 3:
        print("Usage: python test_script.py <save_location>")
        sys.exit(1)

    save_location = sys.argv[2]
    os.makedirs(save_location, exist_ok=True)
    output_file = os.path.join(save_location, 'output.txt')

    with open(output_file, 'w') as f:
        f.write("This is a test file.")

    print(output_file)

if __name__ == "__main__":
    main()
EOF

# Execute the test Python script using the log wrapper
log python "$PYTHON_SCRIPT" --save_location "$SAVE_LOCATION"

# Check the output
if [ -f "$SAVE_LOCATION/output.txt" ]; then
    echo "Output file is correctly placed in the save location."
else
    echo "Output file is missing in the save location."
fi

if [ -f "$SAVE_LOCATION/logs/$(basename "$LOG_WRAPPER_PATH")" ]; then
    echo "log_wrapper.py is correctly copied to the save location logs folder."
else
    echo "log_wrapper.py is missing in the save location logs folder."
fi

if [ -f "$SAVE_LOCATION/logs/$(basename "$PYTHON_SCRIPT")" ]; then
    echo "test_script.py is correctly copied to the save location logs folder."
else
    echo "test_script.py is missing in the save location logs folder."
fi

if [ -f "$SAVE_LOCATION/logs/execution.log" ]; then
    echo "Execution log file is correctly created in the save location logs folder."
else
    echo "Execution log file is missing in the save location logs folder."
fi
