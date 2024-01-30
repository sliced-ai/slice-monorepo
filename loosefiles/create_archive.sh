# I hate getting github functional here. Using desktop this balls up all files under 1MB so I can use the github desktop. I also like to keep data structures and current working state of my directories. 

#!/bin/bash

# Toggle for creating the large archive (set to 1 to enable, 0 to disable)
create_large_archive=0

# Define the root directory to search in
root_dir="/home/ec2-user/environment"
# Define the directory to save archives
archive_dir="$root_dir/archives"
# Create the archive directory if it does not exist
mkdir -p "$archive_dir"
# Define the archive names with date
date_stamp=$(date +%Y-%m-%d)
archive_name="filtered_archive_$date_stamp.tar.gz"
large_archive_name="large_archive_$date_stamp.tar.gz"

# Extensions to ignore (e.g., ".log .tmp .bak")
ignore_extensions="log tmp bak jsonl"

# Navigate to the root directory
cd "$root_dir"

# Output the current directory and list all files
echo "Current directory: $(pwd)"
echo "Listing all files and directories in the current directory:"
ls -lah

# Function to create an archive
create_archive() {
    local file_list="$1"
    local archive_path="$2"

    if [ -z "$file_list" ]; then
        echo "No files found for archive: $(basename $archive_path)"
        return
    fi

    echo "Creating archive: $(basename $archive_path)"
    echo "$file_list" | tr '\n' '\0' | tar --null -czvf "$archive_path" -T -
}

# Find files under 1MB and exclude the specified extensions
echo "Finding files under 1MB (excluding specified extensions)..."
filtered_file_list=$(find . -type f -size -10M ! -path "./archives/*" \
    $(printf "! -name *.%s " $ignore_extensions))

echo "Files to be included in the filtered archive:"
echo "$filtered_file_list"

# Create the filtered archive
create_archive "$filtered_file_list" "$archive_dir/$archive_name"

# Check if large archive creation is enabled
if [ $create_large_archive -eq 1 ]; then
    echo "Finding all files for the large archive..."
    large_file_list=$(find . -type f ! -path "./archives/*")

    echo "Files to be included in the large archive:"
    echo "$large_file_list"

    # Create the large archive
    create_archive "$large_file_list" "$archive_dir/$large_archive_name"
fi

echo "Archiving process completed."




# chmod +x create_archive.sh
# ./create_archive.sh
