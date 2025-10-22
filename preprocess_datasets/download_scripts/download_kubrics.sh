#!/bin/bash

# Check if data_root argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <data_root>"
    echo "Example: $0 /path/to/data"
    exit 1
fi

DATA_ROOT="$1"
TARGET_DIR="${DATA_ROOT}/kubric4d/kubric4d_raw"

# Create target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Save current directory to return to later
ORIGINAL_DIR=$(pwd)

# Change to target directory for downloading
cd "$TARGET_DIR"

BASEURL="https://s3.us-east-1.amazonaws.com/tri-ml-public.s3.amazonaws.com/urp/datasets/tcow"
DLCMD="wget --continue --retry-connrefused --waitretry=10 --timeout=120 --tries=10 --no-check-certificate --show-progress"

$DLCMD $BASEURL/gcd_kubric4d_train.tar.gz.aa
$DLCMD $BASEURL/gcd_kubric4d_train.tar.gz.ab
$DLCMD $BASEURL/gcd_kubric4d_train.tar.gz.ac
$DLCMD $BASEURL/gcd_kubric4d_train.tar.gz.ad
$DLCMD $BASEURL/gcd_kubric4d_train.tar.gz.ae
$DLCMD $BASEURL/gcd_kubric4d_train.tar.gz.af
$DLCMD $BASEURL/gcd_kubric4d_train.tar.gz.ag
$DLCMD $BASEURL/gcd_kubric4d_train.tar.gz.ah
$DLCMD $BASEURL/gcd_kubric4d_train.tar.gz.ai
$DLCMD $BASEURL/gcd_kubric4d_train.tar.gz.aj
$DLCMD $BASEURL/gcd_kubric4d_train.tar.gz.ak
$DLCMD $BASEURL/gcd_kubric4d_train.tar.gz.al
$DLCMD $BASEURL/gcd_kubric4d_train.tar.gz.am
$DLCMD $BASEURL/gcd_kubric4d_train.tar.gz.an
$DLCMD $BASEURL/gcd_kubric4d_train.tar.gz.ao
$DLCMD $BASEURL/gcd_kubric4d_train.tar.gz.ap
$DLCMD $BASEURL/gcd_kubric4d_train.tar.gz.aq
$DLCMD $BASEURL/gcd_kubric4d_train.tar.gz.ar

echo "All files have been downloaded successfully!"
echo "Extracting the dataset to $TARGET_DIR..."

# Extract the dataset
cat gcd_kubric4d_train.tar.gz.* | tar xvfz -

# Clean up the downloaded tar.gz files
echo "Cleaning up downloaded archive files..."
rm -f gcd_kubric4d_train.tar.gz.*

echo "Dataset extraction completed successfully!"
echo "Dataset is now available at: $TARGET_DIR"

# Return to original directory
cd "$ORIGINAL_DIR"