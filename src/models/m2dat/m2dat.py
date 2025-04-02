from . portable_m2d import PortableM2D
from timm.models.layers import trunc_normal_
import torch
import torch.nn.functional as F
import itertools

import logging
logger = logging.getLogger(__name__)

class M2dAt(PortableM2D):
    def __init__(self,
                 weight_file,
                 num_classes,
                 finetuning_layers='head', # head, backbone_out, 1_blocks, 2_blocks, ..., 15_blocks, all
                 ref_channel=None,
                 ):
        super().__init__(weight_file, num_classes=527, freeze_embed=False, flat_features=None)
        self.finetuning_layers = finetuning_layers
        self.num_classes = num_classes
        self.ref_channel = ref_channel

        self.head = torch.nn.Linear(self.cfg.feature_d, self.num_classes)
        
        trunc_normal_(self.head.weight, std=2e-5)
        
        modules = [self.backbone.cls_token, self.backbone.pos_embed, self.backbone.patch_embed, self.backbone.pos_drop, self.backbone.patch_drop, self.backbone.norm_pre] # 0-5
        for block in self.backbone.blocks: modules.append(block) # 6-17
        modules.extend([self.backbone.norm, self.backbone.fc_norm, self.backbone.head_drop]) # 18-20
        modules.extend([self.head_norm, self.head]) # 21-22
        self.md = modules

        finetuning_modules_idx = {
            'head': 21, # modules ~18 is frozen
            'backbone_out': 18,
            'all': 0,
        }
        for i in range(1, len(self.backbone.blocks) + 1):
            finetuning_modules_idx[f'{i}_blocks'] = 17 - i + 1
        self.finetuning_modules_idx = finetuning_modules_idx
        if self.finetuning_layers in finetuning_modules_idx.keys():
            logger.info(f'finetuning: {self.finetuning_layers}')
            modules_idx = finetuning_modules_idx[self.finetuning_layers]
            for i, module in enumerate(modules):
                self._set_requires_grad(module, i >= modules_idx) # from modules_idx, set requires grad == True
        else:
            raise NotImplementedError(f"finetuning_layers mode '{self.finetuning_layers}' has not been implemented")
    
    
    def _set_requires_grad(self, model, requiregrad):
        if isinstance(model, torch.nn.parameter.Parameter): model.requires_grad = requiregrad
        else:
            for param in model.parameters(): param.requires_grad = requiregrad

    
    # copy from PortableM2D, change input output to dict
    # def forward(self, batch_audio, average_per_time_frame=False):
    def forward(self, input_dict):
        batch_audio = input_dict['waveform'] # [bs, wlen] or [bs, nch, wlen]
        if batch_audio.dim() == 3:
            assert self.ref_channel is not None
            batch_audio = batch_audio[:, self.ref_channel, :] # [bs, wlen]
        x = self.encode(batch_audio, average_per_time_frame=False)
        x = x.mean(1)  # B, D
        x = self.head_norm(x.unsqueeze(-1)).squeeze(-1)
        x = self.head(x)
        
        x = torch.sigmoid(x)

        output_dict = {'probabilities': x}
        return output_dict

    def _get_top_prob(self, prob_vector, pthre, nevent_range):
        # prob_vector [bs, nclasses]
        batch_size, n_classes = prob_vector.shape
        onehots = torch.zeros_like(prob_vector)
    
        # Get the top probabilities and their indices
        for i in range(batch_size):
            probs = prob_vector[i]
            selected_indices = (probs >= pthre).nonzero(as_tuple=True)[0]
    
            if len(selected_indices) < nevent_range[0]: # Select at least nevent_range[0] with highest probability
                top_indices = torch.argsort(probs, descending=True)[:nevent_range[0]]
            else: # Select at most nevent_range[1] from those exceeding threshold
                top_indices = selected_indices[torch.argsort(probs[selected_indices], descending=True)[:nevent_range[1]]]
            onehots[i, top_indices] = 1.
        return onehots

    def predict(self, input_dict, pthre=0.5, nevent_range=[1, 3]): # remove the above function later
        if input_dict['waveform'].dim() == 3: # [bs, nch, wlen]
            input_dict['waveform'] = input_dict['waveform'][:, 0, :]
        output_dict = self.forward(input_dict)
        output_dict['multihot_vector'] = self._get_top_prob(output_dict['probabilities'], pthre, nevent_range)
        return output_dict



