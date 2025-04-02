import glob
import os
import argparse
import json
from src.utils import LABELS
all_labels = set(LABELS['dcase2025t4'])

def verify_source_structure(source_dir):
    print(f'\nVerify source directory: {os.path.abspath(source_dir)}', flush=True)

    assert os.path.isdir(source_dir), f'Not a directory: {source_dir}'
    assert os.path.isdir(os.path.join(source_dir, 'src')), f'Missing folder: src'
    print('src: OK', flush=True)

    assert os.path.isfile(os.path.join(source_dir, 'checkpoint/m2d_as_vit_base-80x1001p16x16p32k-240413_AS-FT_enconly/weights_ep69it3124-0.47998.pth')), f'Missing checkpoint: checkpoint/m2d_as_vit_base-80x1001p16x16p32k-240413_AS-FT_enconly/weights_ep69it3124-0.47998.pth'
    print('M2D checkpoint: OK', flush=True)

    assert os.path.isdir(os.path.join(source_dir, 'data')), f'Missing folder: data'
    datasubdirs = [
        'dev_set',
        'dev_set/config',
        'dev_set/interference',
        'dev_set/interference/train',
        'dev_set/interference/valid',
        'dev_set/noise',
        'dev_set/noise/train',
        'dev_set/noise/valid',
        'dev_set/room_ir',
        'dev_set/room_ir/train',
        'dev_set/room_ir/valid',
        'dev_set/sound_event',
        'dev_set/sound_event/train',
        'dev_set/sound_event/valid',
        'dev_set/test',
        'dev_set/test/soundscape',
        'dev_set/test/oracle_target',
    ]
    for subdir in datasubdirs:
        folder = os.path.join(source_dir, 'data', subdir)
        assert os.path.isdir(folder), f'Missing folder: {folder}'
    print('data folders: OK', flush=True)
    print("Source directory verified successfully.", flush=True)
    print()

def verify_data(source_dir):
    missing = []
    filelist_path = os.path.join(source_dir, 'data/dev_set/config/filelist.json')
    assert os.path.isfile(filelist_path), f'Missing filelist: {filelist_path}'
    with open(filelist_path) as f:
        filelist = json.load(f);
        missing = [f for f in filelist if not os.path.isfile(os.path.join(projectfolder, f))]
    if missing:
        print('- ' +  '\n- '.join(missing))
        print(f'Missing {len(missing)} files')
    else:
        print("Data directory verified successfully.")

def verify_soundscape_estimate(soundscape_dir, estimate_dir):
    print('\nVerify soundscape and estimate directories', flush=True)
    soundscape_names = [f[:-4] for f in os.listdir(soundscape_dir) if f.endswith(".wav")]
    soundscape_names = sorted(soundscape_names)
    all_estimate = [f for f in os.listdir(estimate_dir) if f.endswith(".wav")]

    print(f'{len(soundscape_names)} soundscapes in {soundscape_dir}', flush=True)
    print(f'{len(all_estimate)} estimates in {estimate_dir}', flush=True)
    print()
    processed = []
    no_estimate = []
    for soundscape_name in soundscape_names:
        estimates = [e for e in all_estimate if e.startswith(soundscape_name) and e.endswith('.wav')]
        labels = [e[len(soundscape_name) + 1 : -4] for e in estimates]
        processed.extend(estimates)
        for lb, fname in zip(labels, estimates):
            assert lb in all_labels, f'"{fname}" is not a valid filename of the estimates for {soundscape_name}'
        if not estimates: no_estimate.append(soundscape_name)

    if no_estimate:
        print(f'Warning: {len(no_estimate)} soundscapes have no estimate:', flush=True)
        print('- ' + '\n- '.join(no_estimate), flush=True)

    remaining = list(set(all_estimate) - set(processed))
    if remaining:
        print(f'Warning: {len(remaining)} estimates do not belong to any soundscape:', flush=True)
        print('- ' + '\n- '.join(remaining), flush=True)
    print("Soundscape and estimate directories verified successfully.", flush=True)
    print()
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=str, required=False, default='')
    
    parser.add_argument("--soundscape_data_dir", type=str, required=False, default='')
    parser.add_argument("--estimate_data_dir", type=str, required=False, default='')
    args = parser.parse_args()
    
    if args.source_dir: # verify structure of the source directory
        verify_source_structure(args.source_dir)
        
    assert bool(args.soundscape_data_dir) == bool(args.estimate_data_dir), f'Arguments soundscape_data_dir and estimate_data_dir must be provided at the same time.\nOnly {"soundscape_data_dir" if bool(args.soundscape_data_dir) else "estimate_data_dir"} was provided.'
    if bool(args.soundscape_data_dir) and bool(args.estimate_data_dir):
        verify_soundscape_estimate(args.soundscape_data_dir, args.estimate_data_dir)
    print(f'-----Finish-----')
    
    








