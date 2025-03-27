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
                 filelist, # dict or string
                 n_sources,
                 label_set, # 'eusipco25' key of LABELS in utils
                 label_vector_mode='multihot', # multihot, concat, stack
                ):
        super().__init__()
        self.label_set = label_set
        self.filelist = filelist
        self.base_datadir = os.path.dirname(filelist)
        self.n_sources = n_sources
        self.label_vector_mode = label_vector_mode

        with open(self.filelist) as f:
            self.data = json.load(f);

        self.labels = LABELS[self.label_set]
        self.onehots = torch.eye(len(self.labels), requires_grad=False).to(torch.float32)
        self.label_onehots = {label: self.onehots[idx] for idx, label in enumerate(self.labels)}
        self.label_onehots['silence'] = torch.zeros(self.onehots.size(1), requires_grad=False,  dtype=torch.float32)

        self.collate_fn = collate_fn
    
    def get_onehot(self, label):
        return self.label_onehots[label]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        info = self.data[idx]
        mixture_path = os.path.join(self.base_datadir, info['mixture'])
        mixture, _ = librosa.load(mixture_path, sr = None, mono=False)
        labels = list(info['labels'])
        dry_sources = []
        for source_path in info['sources']:
            dry_source, _ = librosa.load(os.path.join(self.base_datadir, source_path), sr = None)
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
            
        item = {
            'mixture': torch.from_numpy(mixture).to(torch.float32), # [nch, wlen]
            'dry_sources': torch.from_numpy(np.stack(dry_sources))[:, None, :].to(torch.float32), # [nevents, 1, wlen]
            'label_vector': label_vector_all, # [nevent, nclass], [nclass], or [nevent x nclass]
            'label': labels, # list
        }
        return item





