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
from src.utils import LABELS, initialize_config, ca_metric

all_labels = LABELS['dcase2025t4']

class Evaluator:
    def __init__(self,
                 config_path,
                 generate_waveform = True,
                 use_generated_waveform=False,
                 batch_size=2):
        self.config_path = config_path
        self.batch_size = batch_size
        self.generate_waveform = generate_waveform
        self.use_generated_waveform = use_generated_waveform
        assert not self.generate_waveform or not self.use_generated_waveform, 'if use_generated_waveform is True, waveform will not be generated again (generate_waveform should be False)'

        with open(self.config_path) as f: config = yaml.safe_load(f)

        dsconfig = config['dataset']

        if not self.use_generated_waveform:
            outputdir = dsconfig['args'].pop('estimate_target_dir', None)
        if self.generate_waveform:
            self.outputdir = outputdir
            os.makedirs(self.outputdir, exist_ok=True)

        dataset = initialize_config(config['dataset'], reload=True)

        # load model and dataset
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                collate_fn=dataset.collate_fn,
                                num_workers=batch_size*2)
        model = initialize_config(config['model'], reload=True)
        model.eval(); model = model.to('cuda')

        self.dataset = dataset
        self.sr = self.dataset.sr
        self.dataloader = dataloader
        self.model = model

    def write_audio(self, batch_est_labels, batch_est_waveforms, batch_soundscape_names):
        for labels, waveforms, soundscape_name in zip(batch_est_labels, batch_est_waveforms, batch_soundscape_names):
            for label, waveform in zip(labels, waveforms):
                if label != 'silence':
                    wavpath = os.path.join(self.outputdir, soundscape_name + '_' + label + '.wav')
                    sf.write(wavpath, waveform.numpy(), self.sr)

    def predict(self, mixture, labels=None):
        mixture = mixture.to('cuda')
        if labels is not None:
            with torch.no_grad():
                batch_est_labels = labels
                output = model.separate(mixture, batch_est_labels)
                batch_est_waveforms = output['waveform'].cpu()[:, :, 0, :]# [bs, nsources, wlen]
        else:
            with torch.no_grad():
                output = self.model.predict_label_separate(mixture, pthre=0.5, nevent_range=[1, 3])
                batch_est_labels = output['label'] # bs, nsources
                batch_est_waveforms = output['waveform'].cpu()[:, :, 0, :]# [bs, nsources, wlen]
        return batch_est_waveforms, batch_est_labels

    def evaluate(self):
        metrics = []
        label_checks = []
        for batch in tqdm(self.dataloader):

            if self.use_generated_waveform:
                batch_est_waveforms = batch['est_dry_sources'][:, :, 0, :]
                batch_est_labels = batch['est_label']
            else:
                batch_est_waveforms, batch_est_labels = self.predict(batch['mixture'])

            if self.generate_waveform:
                os.makedirs(self.outputdir, exist_ok=True)
                self.write_audio(batch_est_labels, batch_est_waveforms, batch['soundscape_name'])

            batch_mixture = batch['mixture'][:, 0, :] # [bs, wlen]
            batch_ref_waveforms = batch['dry_sources'][:, :, 0, :] # [bs, nsources, wlen]
            batch_ref_labels = batch['label']
            for est_lb, est_wf, ref_lb, ref_wf, mixture in zip(batch_est_labels,
                                                                batch_est_waveforms,
                                                                batch_ref_labels,
                                                                batch_ref_waveforms,
                                                                batch_mixture,
                                                                ):
                metric = ca_metric(est_lb, est_wf, ref_lb, ref_wf, mixture, snr)
                label_check = (set(est_lb)- {'silence'}) == (set(ref_lb)- {'silence'})
                metrics.append(metric)
                label_checks.append(label_check)
        print('CA-SDRi: %.3f'%(np.mean(metrics)))
        print('Label prediction accuracy: %.2f'%(np.sum(label_checks)*100/len(label_checks)))

def main(args):
    evalobj = Evaluator(
                 args.config,
                 args.generate_waveform,
                 args.use_generated_waveform,
                 args.batchsize)
    evalobj.evaluate()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True,)
    parser.add_argument("--generate_waveform", action="store_true")
    parser.add_argument("--use_generated_waveform", action="store_true")
    parser.add_argument("--batchsize","-b", type=int, required=False, default=2)

    args = parser.parse_args()
    print('START')
    main(args)

# python -m src.evaluation.evaluate -c src/evaluation/eval_configs/m2d_resunetk.yaml --generate_waveform
