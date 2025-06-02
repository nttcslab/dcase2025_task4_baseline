import random
import numpy as np
import yaml
import json
from tqdm import tqdm
import torch
from torchmetrics.functional import(
    scale_invariant_signal_noise_ratio as si_snr,
    signal_noise_ratio as snr)
import os, sys;
import argparse
import soundfile as sf

from torch.utils.data import DataLoader
from src.utils import LABELS, initialize_config, ca_metric, ignore_warnings
ignore_warnings()
from .evaluate import Evaluator
from src.datamodules.dataset.s5.dataset_s5_waveform import DatasetS5Waveform

all_labels = set(LABELS['dcase2025t4'])


def write_audio(outputdir, sr, batch_est_labels, batch_est_waveforms, batch_soundscape_names):
    for labels, waveforms, soundscape_name in zip(batch_est_labels, batch_est_waveforms, batch_soundscape_names):
        for label, waveform in zip(labels, waveforms):
            if label != 'silence':
                wavpath = os.path.join(outputdir, soundscape_name + '_' + label + '.wav')
                sf.write(wavpath, waveform.numpy(), sr)

def verify(soundscape_dir, output_dir):
    soundscapes = [f[:-4] for f in os.listdir(soundscape_dir) if f.endswith(".wav")]
    est_waveforms = [f[:-4] for f in os.listdir(output_dir) if f.endswith(".wav")]

    for soundscape in soundscapes:
        est_labels = [e[len(soundscape) + 1 :] for e in est_waveforms if e.startswith(soundscape)]
        assert est_labels, f'There is no estimate for "{soundscape}"'
        for e in est_labels:
            assert e in all_labels, f'"{e}" is not a valid label'

def main(args):
    sr = 32000
    # Create output dir
    submission_foldername = args.output_name
    output_dir = os.path.join(args.output_dir, submission_foldername, 'eval_out')
    assert not os.path.isdir(output_dir), f'{output_dir} exists!!'
    os.makedirs(output_dir)

    # Load dataset
    dataset = DatasetS5Waveform(
                 args.soundscape_dir,
                 oracle_target_dir=None,
                 estimate_target_dir=None,
                 n_sources=3,
                 label_set='dcase2025t4', # key of LABELS in utils
                 label_vector_mode=None, # will not be used
                 sr=sr,
                )
    dataloader = DataLoader(dataset,
                            batch_size=args.batchsize,
                            shuffle=False,
                            collate_fn=dataset.collate_fn,
                            num_workers=args.batchsize)

    # Load models
    with open(args.config) as f: config = yaml.safe_load(f)
    model = initialize_config(config['model'], reload=True)
    model.eval(); model = model.to('cuda')

    for batch in tqdm(dataloader):
        with torch.no_grad():
            output = model.predict_label_separate(batch['mixture'].to('cuda'))
            batch_est_labels = output['label'] # bs, nsources
            batch_est_waveforms = output['waveform'].cpu()[:, :, 0, :]# [bs, nsources,wlen]
        write_audio(output_dir, sr, batch_est_labels, batch_est_waveforms, batch['soundscape_name'])

    verify(args.soundscape_dir, output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batchsize","-b", type=int, required=False, default=2)
    parser.add_argument("--config", "-c", type=str, required=True,)

    parser.add_argument("--soundscape_dir", type=str, required=True,)

    parser.add_argument("--output_dir", type=str, required=True,)
    parser.add_argument("--output_name", type=str, required=True,)

    args = parser.parse_args()
    print('START')
    main(args)

# python -m src.evaluation.generate_waveform -c src/evaluation/gen_wav_configs/m2d_resunetk.yaml --soundscape_dir data_/dev_set/test/soundscape --output_dir data_/dev_set/test  --output_name Nguyen_NTT_task4_1_out
