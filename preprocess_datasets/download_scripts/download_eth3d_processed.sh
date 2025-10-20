#!/bin/bash

# This script downloads preprocessed ETH3D dataset required for recreating UFM benchmarks.
# This avoids hours of downloading and preprocessing, allowing quick verification of results.
# If you are interested in developing on top of these benchmarks, you should run the individual dataset download and preprocessing scripts.

if [ -z "$1" ]
then
   echo "Please pass a target directory to this script, e.g.: ./download_all_preprocessed.sh /path/to/data_folder_root";
   echo "This will download eth3d_processed.zip and dtu_processed.zip to the subfolder of the root.";
   exit 1
fi

set -e  # stop on error
TARGET_DIR="$1"

# Create target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Convert to absolute path to avoid issues
TARGET_DIR="$(cd "$TARGET_DIR" && pwd)"

echo "Downloading preprocessed datasets to $TARGET_DIR"

# ETH3D backup
ETH3D_ZIP="$TARGET_DIR/eth3d_processed.zip"
if [ ! -f "$ETH3D_ZIP" ]; then
    echo "Downloading ETH3D preprocessed dataset..."
    gdown 1x7apkg0797OZPOJfiGIraeUxWKH9AF03 -O "$ETH3D_ZIP"
else
    echo "ETH3D zip file already exists, skipping download."
fi

echo "Extracting ETH3D dataset..."
unzip -q "$ETH3D_ZIP" -d "$TARGET_DIR"
rm -f "$ETH3D_ZIP"
echo "ETH3D dataset extracted successfully!"