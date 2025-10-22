#!/bin/bash
# Auto-setup FlyingThings3D dataset; safe to run from anywhere.

if [ -z "$1" ]
then
   echo "Please pass a data root path to this script, e.g.: ./download_flyingthings.sh /absolute/path/to/data";
   echo "The script will create a 'FlyingThings3D' subfolder in the data root and download FlyingThings3D there.";
   exit 1
fi

set -e  # stop on error
DATA_ROOT="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create data root directory if it doesn't exist
mkdir -p "$DATA_ROOT"

# Convert to absolute path to avoid issues
DATA_ROOT="$(cd "$DATA_ROOT" && pwd)"

# Create FlyingThings3D subfolder
TARGET_DIR="$DATA_ROOT"
mkdir -p "$TARGET_DIR"

# Download with continue option to resume interrupted downloads
wget --no-check-certificate --continue -P "$TARGET_DIR" https://lmb.informatik.uni-freiburg.de/data/FlyingChairs/FlyingChairs.zip

# Extract only if not already extracted
if [ ! -d "$TARGET_DIR/FlyingChairs" ]; then
    echo "Extracting FlyingChairs..."
    unzip -q "$TARGET_DIR/FlyingChairs.zip" -d "$TARGET_DIR"
    mv "$TARGET_DIR/FlyingChairs_release" "$TARGET_DIR/FlyingChairs"
else
    echo "FlyingChairs already extracted, skipping..."
fi

