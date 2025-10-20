#!/bin/bash
# Auto-setup Sintel dataset; safe to run from anywhere.

if [ -z "$1" ]
then
   echo "Please pass a data root path to this script, e.g.: ./download_sintel.sh /absolute/path/to/data";
   echo "The script will create a 'Sintel' subfolder in the data root and download Sintel there.";
   exit 1
fi

set -e  # stop on error
DATA_ROOT="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create data root directory if it doesn't exist
mkdir -p "$DATA_ROOT"

# Convert to absolute path to avoid issues
DATA_ROOT="$(cd "$DATA_ROOT" && pwd)"

# Create Sintel subfolder
TARGET_DIR="$DATA_ROOT/Sintel"
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

SINTEL_DIR="$TARGET_DIR"
SINTEL_ZIP="$TARGET_DIR/MPI-Sintel-complete.zip"
SINTEL_URL="http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip"
SINTEL_MD5="2d2836c2c6b4fb6c9d2d2d58189eb014"

# Check if Sintel dataset already exists (look for training folder which should be present after extraction)
if [ ! -d "$SINTEL_DIR/training" ]; then
    [ ! -f "$SINTEL_ZIP" ] && download_if_missing "$SINTEL_URL" "$SINTEL_ZIP"

    echo "Verifying Sintel checksum..."
    if command -v md5sum >/dev/null; then md5=$(md5sum "$SINTEL_ZIP" | awk '{print $1}')
    else md5=$(md5 -q "$SINTEL_ZIP"); fi
    [ "$md5" != "$SINTEL_MD5" ] && { echo "Sintel MD5 mismatch! Re-download."; rm -f "$SINTEL_ZIP"; exit 1; }

    echo "Checksum OK. Extracting Sintel..."
    unzip -q "$SINTEL_ZIP" -d "$TARGET_DIR"
    echo "Sintel ready at $TARGET_DIR."
    # rm -f "$SINTEL_ZIP"
else
    echo "Sintel dataset already prepared at $TARGET_DIR."
fi