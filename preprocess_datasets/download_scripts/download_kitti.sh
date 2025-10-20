#!/bin/bash
# Auto-setup KITTI dataset; safe to run from anywhere.

if [ -z "$1" ]
then
   echo "Please pass a data root path to this script, e.g.: ./download_kitti.sh /absolute/path/to/data";
   echo "The script will create a 'KITTI' subfolder in the data root and download KITTI there.";
   exit 1
fi

set -e  # stop on error
DATA_ROOT="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create data root directory if it doesn't exist
mkdir -p "$DATA_ROOT"

# Convert to absolute path to avoid issues
DATA_ROOT="$(cd "$DATA_ROOT" && pwd)"

# Create KITTI subfolder
TARGET_DIR="$DATA_ROOT/KITTI"
mkdir -p "$TARGET_DIR"

# Helper: download if missing
download_if_missing() {
    local url="$1"
    local dest="$2"
    echo "Downloading $(basename "$dest") ..."
    if command -v wget >/dev/null; then
        wget -q --show-progress -O "$dest" "$url"
    elif command -v curl >/dev/null; then
        curl -L -o "$dest" "$url"
    else
        echo "Error: neither wget nor curl found. Please install one."
        exit 1
    fi
}

KITTI_DIR="$TARGET_DIR"
KITTI_ZIP="$TARGET_DIR/data_scene_flow.zip"
KITTI_URL="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow.zip"
KITTI_MD5="7fa1057aeb2c3e55dbdca8c0400b8a13"

# Check if KITTI dataset already exists (look for training folder which should be present after extraction)
if [ ! -d "$KITTI_DIR/training" ]; then
    [ ! -f "$KITTI_ZIP" ] && download_if_missing "$KITTI_URL" "$KITTI_ZIP"

    echo "Verifying KITTI checksum..."
    if command -v md5sum >/dev/null; then md5=$(md5sum "$KITTI_ZIP" | awk '{print $1}')
    else md5=$(md5 -q "$KITTI_ZIP"); fi
    [ "$md5" != "$KITTI_MD5" ] && { echo "KITTI MD5 mismatch! Re-download."; rm -f "$KITTI_ZIP"; exit 1; }

    echo "Checksum OK. Extracting KITTI..."
    unzip -q "$KITTI_ZIP" -d "$TARGET_DIR"
    echo "KITTI ready at $TARGET_DIR."
    # rm -f "$KITTI_ZIP"
else
    echo "KITTI dataset already prepared at $TARGET_DIR."
fi
