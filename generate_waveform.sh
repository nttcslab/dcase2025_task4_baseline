#!/bin/bash

eval set -- "$(getopt -o "" \
--long config:,output_dir:,author:,affiliation:,submission_number: \
-- "$@")"

while true; do
  case "$1" in
    --config) CONFIG_PATH="$2"; shift 2 ;; # "src/evaluation/gen_wav_configs/m2d_resunetk.yaml"
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;; # data/eval_set
    --author) AUTHOR="$2"; shift 2 ;; # Nguyen
    --affiliation) AFFILIATION="$2"; shift 2 ;; # NTT
    --submission_number) SUBMISSION_NUMBER="$2"; shift 2 ;; # 1
    --) shift; break ;;
    *) break ;;
  esac
done

# Configuration variables
EVAL_SET_DIR="data/eval_set"
SOUNDSCAPE_DIR="$EVAL_SET_DIR/soundscape"


OUTPUT_FOLDER_NAME="${AUTHOR}_${AFFILIATION}_task4_${SUBMISSION_NUMBER}_out"
ZIP_NAME="${OUTPUT_FOLDER_NAME}.zip"
OUTPUT_FOLDER="${EVAL_SET_DIR}/${OUTPUT_FOLDER_NAME}"

# Generate waveform
python -m src.evaluation.generate_waveform \
  -c "$CONFIG_PATH" \
  --soundscape_dir "$SOUNDSCAPE_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --output_name "$OUTPUT_FOLDER_NAME"

# Create zip file for submission
if [ -d "$OUTPUT_FOLDER" ]; then
  echo "Zipping: $OUTPUT_FOLDER_NAME"
  cd "${EVAL_SET_DIR}"
  zip -r "$ZIP_NAME" "$OUTPUT_FOLDER_NAME"
  cd -
  echo "Zipped to $ZIP_NAME"
else
  echo "Error: Output folder $OUTPUT_FOLDER not found."
  exit 1
fi
