#!/usr/bin/env sh
# This script converts the data into lmdb/leveldb format,
# depending on the value assigned to $BACKEND.
set -e

EXAMPLE="/home/tim/datasets/cifar10"
DATA="/home/tim/datasets/cifar10"
BUILD=$CAFFE_ROOT/build/tools

# RGB image or GRAY image
GRAY=false

RESIZE=false
if $RESIZE; then
  RESIZE_HEIGHT=32
  RESIZE_WIDTH=32
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

BACKEND="lmdb"

echo "Creating ${BACKEND}..."

rm -rf $EXAMPLE/train_${BACKEND}
rm -rf $EXAMPLE/test_${BACKEND}

$BUILD/convert_imageset -backend=$BACKEND -gray=$GRAY -resize_width=$RESIZE_WIDTH -resize_height=$RESIZE_HEIGHT -shuffle=true $DATA/ $DATA/train.txt  $EXAMPLE/train_${BACKEND}
$BUILD/convert_imageset -backend=$BACKEND -gray=$GRAY -resize_width=$RESIZE_WIDTH -resize_height=$RESIZE_HEIGHT -shuffle=true $DATA/ $DATA/test.txt  $EXAMPLE/test_${BACKEND}

echo "Computing image mean..."

$BUILD/compute_image_mean -backend=$BACKEND \
  $EXAMPLE/train_${BACKEND} $EXAMPLE/mean.binaryproto 2>&1 | tee $EXAMPLE/mean.log

echo "Done."
