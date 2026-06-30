#!/bin/bash

# Usage: bash run_baseline.sh <method>
# where <method> is one of: art, groot, flirt
#
# Configure CONDA_ENV, TARGET_NAME, and BASE_FOLDER below.

set -euo pipefail

METHOD="${1:-art}"

# --- Configuration ---
CONDA_ENV="apex"
TARGET_NAME="safe-sd-v2-1"
BASE_FOLDER="./output"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

if [ -f .env ]; then
    export $(cat .env | xargs)
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "method: $METHOD"
echo "target: $TARGET_NAME"
echo "output: $BASE_FOLDER"

python run_baseline.py \
    --method "$METHOD" \
    --target_name "$TARGET_NAME" \
    --base_folder "$BASE_FOLDER" \
    --judge_devices cuda:0 cuda:1 \
    --method_devices cuda:2 cuda:3
