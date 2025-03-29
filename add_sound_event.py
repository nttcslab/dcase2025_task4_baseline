import os
import argparse
import csv
import shutil
from tqdm import tqdm
import json
import librosa
import soundfile as sf
import numpy as np

FILE_NAME = "file_name"
T4_LABEL = "t4_label"
CONVERT_PATH = "convert_path"

from add_interference import process_wav, show_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--pickup_json", type=str)
    parser.add_argument("--target_sr", type=int, default=32000)
    parser.add_argument("--amp_threshold", type=float, default=0.02)
    parser.add_argument("--min_length", type=float, default=0.1)
    parser.add_argument("--segment", type=float, default=10)
    parser.add_argument("--shift", type=float, default=0.1)
    parser.add_argument("--info_outpath", type=str, default='')
    args = parser.parse_args()

    config = {
        "target_sr": args.target_sr,
        "amp_threshold": args.amp_threshold,
        "min_length": args.min_length,
        "segment": args.segment,
        "shift": args.shift,
        "info_outpath": args.info_outpath,
    }
    
    print('====================================================')           
    print('input_dir    :', args.input_dir)
    print('output_dir   :', args.output_dir)
    print('pickup_json   :', args.pickup_json)
    print('output_sr    :', config['target_sr'])
    print('amp_threshold:', config['amp_threshold'])
    print('min_length   :', config['min_length'])
    print('segment      :', config['segment'])
    print('shift        :', config['shift'])
    if args.info_outpath:
        print('info_outpath :', config['info_outpath'])
    print('====================================================') 
    
    os.makedirs(args.output_dir, exist_ok=True)

    infos = []
    with open(args.pickup_json) as f:
        json_info = json.load(f)
        for i, row in enumerate(tqdm(json_info)):
            source_path = os.path.join(
                args.input_dir,
                row[FILE_NAME]
            )
            assert os.path.exists(source_path), f"{source_path} is not found"
            target_path = os.path.join(
                args.output_dir,
                row[CONVERT_PATH]
            )
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            assert not os.path.exists(target_path), f"{target_path} is exists"
            
            info = process_wav(source_path, target_path, config)
            infos.append(info)
            
    show_info(infos, config['info_outpath'], os.path.basename(args.input_dir))