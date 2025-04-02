import os
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import re
import numpy as np
import struct
import librosa
import random
import warnings

from src.utils import LABELS

def collate_fn(list_data_dict):
    data = {k: [] for k in list_data_dict[0].keys()}
    for ddict in list_data_dict:
        for k in data:
            data[k].append(ddict[k])
    for k in data.keys():
        if type(data[k][0]) is torch.Tensor:
            data[k] = torch.stack(data[k], 0)
    return data



class DatasetS5Waveform(torch.utils.data.Dataset):
    def __init__(self,
                 soundscape_dir, # dict or string
                 oracle_target_dir=None,
                 estimate_target_dir=None,
                 n_sources=3,
                 label_set='dcase2025t4', # key of LABELS in utils
                 label_vector_mode='concat', # multihot, concat, stack
                 sr=32000,

                ):
        super().__init__()
        self.sr = sr
        self.soundscape_dir = soundscape_dir
        self.oracle_target_dir = oracle_target_dir
        self.estimate_target_dir = estimate_target_dir
        self.n_sources = n_sources
        self.label_set = label_set
        self.label_vector_mode = label_vector_mode

        self.labels = LABELS[self.label_set]
        self.onehots = torch.eye(len(self.labels), requires_grad=False).to(torch.float32)
        self.label_onehots = {label: self.onehots[idx] for idx, label in enumerate(self.labels)}
        self.label_onehots['silence'] = torch.zeros(self.onehots.size(1), requires_grad=False,  dtype=torch.float32)

        self.data = [{'soundscape': f[:-4],
                      'mixture_path': os.path.join(self.soundscape_dir, f)
                      } for f in os.listdir(self.soundscape_dir) if f.endswith(".wav")]
        self.data = sorted(self.data, key=lambda x: x['soundscape'])

        if self.oracle_target_dir is not None:
            self._get_data(self.data, 'ref', self.oracle_target_dir)

        if self.estimate_target_dir is not None:
            self._get_data(self.data, 'est', self.estimate_target_dir)

        self.collate_fn = collate_fn

    def _get_data(self, data, est_ref, source_dir):
        if not os.path.exists(source_dir):
            raise FileNotFoundError(f"Source directory '{source_dir}' does not exist.")
        all_wav = [f for f in os.listdir(source_dir) if f.endswith(".wav")]
        for d in data:
            sources = [w for w in all_wav if w.startswith(d['soundscape']) and w.endswith('.wav')]
            d[est_ref + '_label'] = [s[len(d['soundscape']) + 1 : -4] for s in sources]
            d[est_ref + '_source_paths'] = [os.path.join(source_dir, s) for s in sources]
            
            if not sources: warnings.warn(f'No estimate for {d["mixture_path"]}')
            for lb, fname in zip(d[est_ref + '_label'], sources):
                assert lb in self.labels, f'"{fname}" is not a valid filename of the estimates for {d["soundscape"]}'

    def get_onehot(self, label):
        return self.label_onehots[label]

    def __len__(self):
        return len(self.data)

    def _get_label_sources(self, info, est_ref):
        labels = list(info[est_ref + '_label'])
        dry_sources = []
        for source_path in info[est_ref + '_source_paths']:
            dry_source, sr = librosa.load(source_path, sr = None)
            assert sr == self.sr, f'sr of {source_path} ({sr}) is different from the target sr ({self.sr})'
            dry_sources.append(dry_source)
        assert len(labels) == len(dry_sources)

        if len(labels) < self.n_sources:
            for _ in range(self.n_sources - len(labels)):
                labels.append('silence')
                dry_sources.append(np.zeros_like(dry_sources[0]))

        label_vector_all = torch.stack([self.get_onehot(label) for label in labels])
        if self.label_vector_mode == 'multihot': label_vector_all = torch.any(label_vector_all.bool(), dim=0).float() # [nclass]
        elif self.label_vector_mode == 'concat': label_vector_all = label_vector_all.flatten() # [nevent x nclass]
        elif self.label_vector_mode == 'stack': pass  # [nevent, nclass]
        else: raise NotImplementedError(f'label_vector_mode of "{self.label_vector_mode}" has not been implemented')

        prefix = '' if est_ref == 'ref' else 'est_'
        item = {
            prefix + 'dry_sources': torch.from_numpy(np.stack(dry_sources))[:, None, :].to(torch.float32), # [nevents, 1, wlen]
            prefix + 'label_vector': label_vector_all, # [nevent, nclass], [nclass], or [nevent x nclass]
            prefix + 'label': labels, # list
        }

        return item

    def __getitem__(self, idx):
        info = self.data[idx]
        mixture, sr = librosa.load(info['mixture_path'], sr = None, mono=False)
        assert sr == self.sr, f'sr of {info["mixture_path"]} ({sr}) is different from the target sr ({self.sr})'
        mixture = torch.from_numpy(mixture).to(torch.float32)
        item = {
            'soundscape_name': info['soundscape'],
            'mixture': mixture, # [nch, wlen]
        }

        if self.oracle_target_dir is not None:
            output = self._get_label_sources(info, 'ref')
            item.update(output)

        if self.estimate_target_dir is not None:
            output = self._get_label_sources(info, 'est')
            item.update(output)
        return item





