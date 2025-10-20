#!/bin/bash

# This script downloads preprocessed DTU dataset required for recreating UFM benchmarks.
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

# DTU backup
DTU_ZIP="$TARGET_DIR/dtu_processed.zip"
if [ ! -f "$DTU_ZIP" ]; then
    echo "Downloading DTU preprocessed dataset..."
    gdown 1U3_J09Podl7FJnGoQxzZwtIlbFN3zS-U -O "$DTU_ZIP"
else
    echo "DTU zip file already exists, skipping download."
fi

echo "Extracting DTU dataset..."
unzip -q "$DTU_ZIP" -d "$TARGET_DIR"
rm -f "$DTU_ZIP"
echo "DTU dataset extracted successfully!"

echo "All downloads and extractions completed successfully!"
echo "Preprocessed datasets are now available in $TARGET_DIR"