#!/bin/bash

# data downloaded from huggingface
SOURCE_DIR="/workspace/slice-monorepo/cl_cr3/pipeline/aligneddata"
DEST_DIR="/workspace/slice-monorepo/cl_cr3/pipeline/cr3_data"

# Create the destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Loop through each subdirectory in the source directory
for dir in "$SOURCE_DIR"/*; do
  if [ -d "$dir" ]; then
    # Find and copy all .json files to the destination directory
    find "$dir" -name "*.json" -exec cp {} "$DEST_DIR" \;
  fi
done

echo "All .json files have been copied to $DEST_DIR"
