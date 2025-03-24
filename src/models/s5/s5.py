import torch
import importlib

from src.utils import LABELS, initialize_config

class S5(torch.nn.Module):
    def __init__(
        self,
        tagger_config,
        separator_config,
        label_set,
        tagger_ckpt=None,
        separator_ckpt=None,
    ):
        super().__init__()

        tagger = initialize_config(tagger_config) # checkpoint loaded
        separator = initialize_config(separator_config)

        if separator_ckpt is not None:
            self._load_ckpt(separator_ckpt, separator)
        if tagger_ckpt is not None:
            self._load_ckpt(tagger_ckpt, tagger)
        
        separator.eval();
        tagger.eval();

        self.tagger = tagger
        self.separator = separator

        self.label_set = label_set
        self.labels = LABELS[self.label_set]
        self.onehots = torch.eye(len(self.labels), requires_grad=False).to(torch.float32)
        self.label_onehots = {label: self.onehots[idx] for idx, label in enumerate(self.labels)}
        self.label_onehots['silence'] = torch.zeros(self.onehots.size(1), requires_grad=False,  dtype=torch.float32)
    
    def _load_ckpt(self, path, model):
        model_ckpt = torch.load(path, weights_only=False, map_location='cpu')['state_dict']
        if set(model.state_dict().keys()) != set(model_ckpt.keys()): # remove prefix, incase the ckpt is of lightning module
            one_model_key = next(iter(model.state_dict().keys()))
            ckpt_corresponding_key = [k for k in model_ckpt.keys() if k.endswith(one_model_key)]
            prefix = ckpt_corresponding_key[0][:-len(one_model_key)]
            model_ckpt = {k[len(prefix):]: v for k, v in model_ckpt.items() if k.startswith(prefix)}# remove prefix
        model.load_state_dict(model_ckpt)  
    
    def _get_label(self, batch_multihot_vecor): # [bs, nclass]
        labels = []
        for multihot in batch_multihot_vecor:
            label = [l for i, l in enumerate(self.labels) if multihot[i]>0]
            labels.append(label)
        return labels # [[], [], ...]
    def predict_label(self, waveforms, pthre=0.5, nevent_range=[1, 3]): # TODO: change to kwards
        output = self.tagger.predict({'waveform': waveforms}, pthre=pthre, nevent_range=nevent_range)
        labels= self._get_label(output['multihot_vector'])
        for i in range(len(labels)):
            if len(labels[i])< nevent_range[1]: labels[i] += ['silence']*(nevent_range[1] - len(labels[i]))
        return {'label': labels}

    def _get_label_vector(self, batch_labels):
        return torch.stack([torch.stack([self.label_onehots[label] for label in labels]).flatten() for labels in batch_labels])
    def separate(self, batch_mixture, batch_labels):
        # mixture [bs, 4, wlen]
        # labels [[lb1, lb2,...], ...   ]
        label_vector = self._get_label_vector(batch_labels).to(batch_mixture.device)
        separator_input = {
            'mixture': batch_mixture, 
            'label_vector': label_vector,
        }
        separator_output = self.separator(separator_input)
        return {'waveform': separator_output['waveform']} # {'waveform': []}
    
    def predict_label_separate(self, mixture, pthre=0.5, nevent_range=[1, 3]):
        # mixtures[bs, 4, wlen]
         # [bs, wlen]
        predict_labels = self.predict_label(mixture, pthre=pthre, nevent_range=nevent_range)
        predict_waveforms = self.separate(mixture, predict_labels['label'])
        reobj = {
            'label': predict_labels['label'],
            'waveform': predict_waveforms['waveform']
        }
        return reobj











