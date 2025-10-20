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

TA_WB_FOLDER="$TARGET_DIR/TA-WB"

# Create target directory if it doesn't exist
mkdir -p "$TA_WB_FOLDER"

# Convert to absolute path to avoid issues
TA_WB_FOLDER="$(cd "$TA_WB_FOLDER" && pwd)"

echo "Downloading preprocessed datasets to $TA_WB_FOLDER"

# TA-WB folder
if [ ! -f "$TA_WB_FOLDER" ]; then
    echo "Downloading TA-WB preprocessed dataset..."
    gdown --folder 1WYQ3WGm9lhbxcMWH1OMJqVdXQvtrZ6w8 -O "$TA_WB_FOLDER"
else
    echo "TA-WB folder already exists, skipping download."
fi
