#!/bin/bash

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --semhear_path)
            SEMHEAR_PATH="$2"
            shift 2
            ;;
        --ears_path)
            EARS_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
done

TARGET_SR=32000
AMP_THRESHOLD=0.02
MIN_LENGTH=0.1
SEGMENT=10
SHIFT=0.1

python add_interference.py \
    --input_dir="${SEMHEAR_PATH}/bg_scaper_fmt" \
    --output_dir=data/dev_set/interference \
    --target_sr=$TARGET_SR \
    --amp_threshold=$AMP_THRESHOLD \
    --min_length=$MIN_LENGTH \
    --segment=$SEGMENT \
    --shift=$SHIFT

python add_sound_event.py \
    --input_dir="${SEMHEAR_PATH}/FSD50K" \
    --output_dir=data/dev_set/sound_event \
    --pickup_json=data/dev_set/config/FSD50K_config.json \
    --target_sr=$TARGET_SR \
    --amp_threshold=$AMP_THRESHOLD \
    --min_length=$MIN_LENGTH \
    --segment=$SEGMENT \
    --shift=$SHIFT

python add_sound_event.py \
    --input_dir="${EARS_PATH}" \
    --output_dir=data/dev_set/sound_event \
    --pickup_json=data/dev_set/config/EARS_config.json \
    --target_sr=$TARGET_SR \
    --amp_threshold=$AMP_THRESHOLD \
    --min_length=$MIN_LENGTH \
    --segment=$SEGMENT \
    --shift=$SHIFT