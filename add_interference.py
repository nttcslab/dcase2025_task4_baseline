import os
import shutil
from tqdm import tqdm
import soundfile as sf
import argparse
import librosa
import numpy as np
import json

selected_classes = ["Air conditioning", "Aircraft", "Bird flight, flapping wings", "Bleat", "Boiling", "Boom",
              "Burping, eructation", "Burst, pop", "Bus", "Camera", "Car passing by",
              "Cattle, bovinae", "Chainsaw", "Chewing, mastication", "Chink, clink", "Clip-clop",
              "Cluck", "Clunk", "Coin (dropping)", "Crack", "Crackle", "Creak", "Croak", "Crow",
              "Crumpling, crinkling", "Crushing", "Drill", "Drip", "Electric toothbrush", "Engine",
              "Fart", "Finger snapping", "Fire", "Fire alarm", "Firecracker", "Fireworks", "Fixed-wing aircraft, airplane",
              "Frog", "Gears", "Growling", "Gurgling", "Helicopter", "Hiss", "Hoot", "Howl", "Howl (wind)",
              "Jackhammer", "Keys jangling", "Lawn mower", "Light engine (high frequency)", "Microwave oven",
              "Moo", "Oink", "Packing tape, duct tape", "Pig", "Printer", "Purr", "Rain", "Rain on surface",
              "Raindrop", "Ratchet, pawl", "Rattle", "Sanding", "Sawing", "Scissors", "Screech", "Sheep",
              "Ship", "Shuffling cards", "Skateboard", "Slam", "Sliding door", "Sneeze", "Sniff", "Snoring",
              "Splinter", "Squeak", "Stream", "Subway, metro, underground", "Tap", "Tearing", "Thump, thud",
              "Tick", "Tick-tock", "Toothbrush", "Traffic noise, roadway noise", "Train", "Train horn",
              "Velcro, hook and loop fastener", "Waterfall", "Whoosh, swoosh, swish", "Wind", "Writing", "Zipper (clothing)"]
    
def process_wav(src_wavpath, dest_wavpath, config):
    info = {
        'src_path': src_wavpath,
        'dest_path': dest_wavpath,
        'status': 'selected', # 'remove_low_amplitude', 'remove_too_short', 'remove_contain_zero_segment', 'remove_too_short_after_trim'
                              # 'selected', 'selected_trim', 'selected_resampled', 'selected_trim_resampled'
    }
    audio, sr = librosa.load(src_wavpath, sr=None)
    assert audio.ndim == 1
    original_length = audio.shape[0]
    
    # Check amplitude
    if max(abs(audio)) < config['amp_threshold']:
        info['status'] = 'remove_low_amplitude'
        return info
    
    # Check length
    if original_length < config['min_length']*sr:
        info['status'] = 'remove_too_short'
        return info
    
    # Trim audio
    audio = np.trim_zeros(audio)
    if audio.shape[0] < int(config['min_length']*sr):
        info['status'] = 'remove_too_short_after_trim'
        return info
    
    # Check zero segments
    segment_samples = int(config['segment']*sr)
    shift_samples = int(config['shift']*sr)
    for start in range(0, len(audio) - segment_samples + 1, shift_samples):
        if np.max(np.abs(audio[start:start + segment_samples])) < config['amp_threshold']:
            info['status'] = 'remove_contain_low_amplitude_segment'
            return info
    
    if audio.shape[0] != original_length:
        info['status'] += '_trim'
    
    if sr != config['target_sr']: # resample if the original sr differs from target sr
        audio = librosa.resample(audio, orig_sr=sr, target_sr=config['target_sr'])
        info['status']+='_resampled'
    
    sf.write(dest_wavpath, audio, config['target_sr'])
    return info

def show_info(infos, info_outpath='', info_name=''):
    print(f"Process {len(infos)} files")
    print(f'Keep {len([f for f in infos if f["status"].startswith("selected")])} files')
    print(f'Remove {len([f for f in infos if f["status"].startswith("remove")])} files')
    all_status = sorted(list(set([f['status'] for f in infos])))
    for status in all_status:
        info_with_status = [f for f in infos if f['status']==status]
        print(f'- {status}: {len(info_with_status)} files')
        if info_outpath:
            os.makedirs(info_outpath, exist_ok=True)
            with open(os.path.join(info_outpath, status +'_' +info_name+'.json'), "w") as json_file:
                json.dump(info_with_status, json_file)
        

def prepare_data(input_dir,
                output_dir,
                selected_classes,
                config,
                ):
    print('input_dir    :', input_dir)
    print('output_dir   :', output_dir)
    
    infos = []
    os.makedirs(output_dir, exist_ok=True)
    for folder in selected_classes:
        src_path = os.path.join(input_dir, folder)
        assert os.path.exists(src_path) and os.path.isdir(src_path), f'There is no {src_path}'
    
    for folder in tqdm(selected_classes):
        src_path = os.path.join(input_dir, folder)
        dest_path = os.path.join(output_dir, folder)
        
        os.makedirs(dest_path, exist_ok=True)
        wavfiles = [f for f in os.listdir(src_path) if f.endswith('.wav')]
        for wavfile in wavfiles:
            src_wavpath = os.path.join(src_path, wavfile)
            dest_wavpath = os.path.join(dest_path, wavfile)
        
            info = process_wav(src_wavpath, dest_wavpath, config)
            infos.append(info)
        
        assert os.path.isdir(dest_path), f"No wav file in {dest_path}\nConsider remove class {folder}"
    show_info(infos, config['info_outpath'], os.path.basename(input_dir))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
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
    print('output_sr    :', config['target_sr'])
    print('amp_threshold:', config['amp_threshold'])
    print('min_length   :', config['min_length'])
    print('segment      :', config['segment'])
    print('shift        :', config['shift'])
    if args.info_outpath:
        print('info_outpath :', config['info_outpath'])
    print('====================================================') 
    
    prepare_data(input_dir= os.path.join(args.input_dir, 'train'),
                 selected_classes=selected_classes,
                 output_dir= os.path.join(args.output_dir, 'train'),
                 config=config)
    prepare_data(input_dir= os.path.join(args.input_dir, 'val'),
                 selected_classes=selected_classes,
                 output_dir= os.path.join(args.output_dir, 'valid'),
                 config=config)
# python interference_sources_prepare.py -i 'data/BinauralCuratedDataset/bg_scaper_fmt' -o 'data/interference'




               
