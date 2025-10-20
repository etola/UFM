#!/bin/bash

# This script downloads metadata required to run the ETH3D and DTU dataset for recreating UFM benchmarks.

if [ -z "$1" ]
then
   echo "Please pass a target directory to this script, e.g.: ./download_pairs_eth3d_dtu.sh /path/to/data_folder_root";
   echo "This will download anymap_data_pairs.zip to the subfolder of the root.";
   exit 1
fi

set -e  # stop on error
TARGET_DIR="$1"

# Create target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Convert to absolute path to avoid issues
TARGET_DIR="$(cd "$TARGET_DIR" && pwd)"

echo "Downloading metadata to $TARGET_DIR"

# AnyMap data pairs
ANYMAP_ZIP="$TARGET_DIR/anymap_data_pairs.zip"
if [ ! -f "$ANYMAP_ZIP" ]; then
    echo "Downloading AnyMap data pairs..."
    gdown 1wXipGzAPQd3hZ91oN4StwRxGEYiq1ow4 -O "$ANYMAP_ZIP"
else
    echo "AnyMap zip file already exists, skipping download."
fi

echo "Extracting AnyMap data pairs..."
unzip -q "$ANYMAP_ZIP" -d "$TARGET_DIR"
rm -f "$ANYMAP_ZIP"
echo "AnyMap data pairs extracted successfully!"

echo "All downloads and extractions completed successfully!"
echo "Metadata is now available in $TARGET_DIR"