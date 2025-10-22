#!/bin/bash
# Auto-setup FlyingThings3D dataset; safe to run from anywhere.

if [ -z "$1" ]
then
   echo "Please pass a data root path to this script, e.g.: ./download_monkaa.sh /absolute/path/to/data";
   echo "The script will create a 'Monkaa' subfolder in the data root and download Monkaa there.";
   exit 1
fi

set -e  # stop on error
DATA_ROOT="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create data root directory if it doesn't exist
mkdir -p "$DATA_ROOT"

# Convert to absolute path to avoid issues
DATA_ROOT="$(cd "$DATA_ROOT" && pwd)"

# Create Monkaa subfolder
TARGET_DIR="$DATA_ROOT/raw_data/monkaa_raw"
mkdir -p "$TARGET_DIR"

# Download with continue option to resume interrupted downloads
wget --no-check-certificate --continue -P "$TARGET_DIR" https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/Monkaa/raw_data/monkaa__frames_cleanpass.tar
wget --no-check-certificate --continue -P "$TARGET_DIR" https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/Monkaa/raw_data/monkaa__frames_finalpass.tar
wget --no-check-certificate --continue -P "$TARGET_DIR" https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/Monkaa/derived_data/monkaa__optical_flow.tar.bz2
wget --no-check-certificate --continue -P "$TARGET_DIR" https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/CameraData_august16/data/Monkaa/raw_data/monkaa__camera_data.tar
wget --no-check-certificate --continue -P "$TARGET_DIR" https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/Monkaa/derived_data/monkaa__disparity.tar.bz2
wget --no-check-certificate --continue -P "$TARGET_DIR" https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/Monkaa/derived_data/monkaa__disparity_change.tar.bz2

