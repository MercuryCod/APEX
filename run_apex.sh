#!/bin/bash

# Usage: bash run_apex.sh
# Configure TARGET_NAME and BASE_FOLDER below.

# --- Configuration ---
TARGET_NAME="safe-sd-v2-1"
BASE_FOLDER="./output"

# Activate uv venv
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.venv/bin/activate"

# Load environment variables from .env file
if [ -f .env ]; then
    export $(cat .env | xargs)
fi

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "target: $TARGET_NAME"
echo "output: $BASE_FOLDER"
echo "environment: $VIRTUAL_ENV"

# Run APEX
# With CUDA_VISIBLE_DEVICES=0,1,...,7, devices are numbered cuda:0 through cuda:7
python main.py --target_name $TARGET_NAME \
        --base_folder $BASE_FOLDER \
        --judge_devices cuda:0 cuda:1 \
        --apex_devices cuda:2 cuda:3 > logs/$TARGET_NAME.log 2>&1 &
