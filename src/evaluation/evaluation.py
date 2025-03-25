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

from torch.utils.data import DataLoader
from src.utils import LABELS, initialize_config, ca_metric

def metric_cal(obj, metric_args, metricfunction, at_name, metricfunctionname):
    obj[f'{at_name}_{metricfunctionname}'] = ca_metric(**metric_args, metricfunc = metricfunction)
def check_label(idict): return (set(idict['est_label'])- {'silence'}) == (set(idict['ref_label'])- {'silence'})
all_labels = LABELS['dcase2025t4']

def main(args):
    with open(args.config) as stream: config = yaml.safe_load(stream)
    dataset = initialize_config(config['dataset'], reload=True)

    # load model and dataset
    dataloader = DataLoader(dataset, batch_size=args.batchsize, shuffle=False, collate_fn=dataset.collate_fn, num_workers=args.batchsize*2)
    model = initialize_config(config['model'], reload=True)
    model.eval(); model = model.to('cuda')
    # outputpath = os.path.join(results_dir, 'resunetk.json')
    
    eval_objs = []
    i_data = -1
    for batch in tqdm(dataloader):
        ref_label = batch['label']
        ref_waveform = batch['dry_sources'] # [bs, nsources, 1, wlen]
        mixture = batch['mixture'].to('cuda')
        assert (model._get_label_vector(ref_label) == batch['label_vector']).all()
        with torch.no_grad():
            est_label_ora = ref_label
            output_ora = model.separate(mixture, est_label_ora)
            est_waveform_ora = output_ora['waveform'].cpu()
            
            both = model.predict_label_separate(mixture, pthre=0.5, nevent_range=[1, 3])
            est_label = both['label']
            est_waveform = both['waveform'].cpu()
    
    
            assert ref_waveform.shape == est_waveform.shape
            assert ref_waveform.shape == est_waveform_ora.shape
            assert ref_waveform.shape[2] == 1 # channel
            mixture = batch['mixture'][:, 0, :] # [bs, wlen]
            ref_waveform = ref_waveform[:, :, 0, :] # [bs, nsources, wlen]
            est_waveform = est_waveform[:, :, 0, :] # [bs, nsources, wlen]
            est_waveform_ora = est_waveform_ora[:, :, 0, :]
            for ib in range(len(est_label)):
                i_data = i_data + 1
                obj = {}
                obj['index'] = i_data
                obj['ref_label'] = ref_label[ib]
                obj[f'est_label'] = est_label[ib]
    
                
                args_est = {'est_lb': est_label[ib], 'est_wf': est_waveform[ib], 'ref_lb': ref_label[ib], 'ref_wf': ref_waveform[ib], 'mixture': mixture[ib]}
                metric_cal(obj, args_est, metricfunction=snr, at_name='est', metricfunctionname='snr')
                metric_cal(obj, args_est, metricfunction=si_snr, at_name='est', metricfunctionname='si_snr')
    
                args_ora = {'est_lb': est_label_ora[ib], 'est_wf': est_waveform_ora[ib], 'ref_lb': ref_label[ib], 'ref_wf': ref_waveform[ib], 'mixture': mixture[ib]}
                metric_cal(obj, args_ora, metricfunction=snr, at_name='ora', metricfunctionname='snr')
                metric_cal(obj, args_ora, metricfunction=si_snr, at_name='ora', metricfunctionname='si_snr')
                
                eval_objs.append(obj)

    data = eval_objs
    print('label prediction accuracy')
    for nevent in [1,2, 3, None]:
        if nevent is not None: data_ = [d for d in data if len(set(d['ref_label']) - {'silence'}) == nevent]
        else: data_ = data
        ac = [check_label(i) for i in data_]
        print('average' if nevent is None else f'{nevent} events', ': %.1f'%(np.sum(ac)*100/len(ac)))

    print('results')
    mdata = {k: [] for k in set(data[0].keys()) - {'index', 'ref_label', 'est_label'}}
    methods = ['est_snr', 'ora_snr', 'est_si_snr', 'ora_si_snr']
    method_names = ['ca_snr', 'ca_snr (oracle labels)', 'ca_si_snr', 'ca_si_snr (oracle labels)']
    for k in mdata.keys():
        mdata[k] = torch.tensor([d[k] for d in data])
    results = torch.stack([mdata[k] for k in methods], dim=1).numpy()
    for name, val in zip(method_names, results.mean(0)):
        print(name, ': ', val)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
    )
    
    parser.add_argument(
        "--batchsize","-b",
        type=int,
        required=False,
        default=2
    )

    args = parser.parse_args()
    print('START')
    main(args)

# python -m src.evaluation.evaluation -c src/evaluation/eval_configs/valid_m2d_resunetk.yaml
