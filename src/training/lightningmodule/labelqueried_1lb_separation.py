from .base_lightningmodule import BaseLightningModule
import random
import torch

class LabelQueriedSeparationLightning1LB(BaseLightningModule):
    def _se_selection(self, batch_labels):
        return [
            random.choice([i for i in range(len(labels)) if labels[i]!='silence'])
            for labels in batch_labels
        ]
    def training_step_processing(self, batch_data_dict, batch_idx):
        """
        process batch_data_dict and return loss

        Returns:
            batchsize (int)
            loss_dict (dict)
                {
                    "loss": loss_val # must have, for back probagation
                    "other_loss_or_metric_name": val, # for logging
                }
        """
        batchsize = batch_data_dict['mixture'].shape[0]
        nsources = batch_data_dict['dry_sources'].shape[1]

        batch_labels = batch_data_dict['label']
        batch_gt = batch_data_dict['dry_sources'] # [bs, nsources, 1 ch, wlen]
        batch_label_vector = batch_data_dict['label_vector'] # [bs, nsources x nclasses]
        batch_label_vector_stack = batch_label_vector.view(batchsize, nsources, batch_label_vector.shape[-1]//nsources)# [bs, nsources, nclasses]

        selected_idx = self._se_selection(batch_labels)
        batch_gt_sel = torch.stack([batch_gt[i, se_idx, :, :] for i, se_idx in enumerate(selected_idx)], dim=0) # [bs, 1 ch, wlen]
        batch_label_vector_sel = torch.stack([batch_label_vector_stack[i, se_idx, :] for i, se_idx in enumerate(selected_idx)], dim=0) # [bs, nclasses]

        input_dict = {
            'mixture': batch_data_dict['mixture'], # [bs, nch, wlen]
            'label_vector': batch_label_vector_sel # [bs, label_len]
        }
        output_dict = self.model(input_dict) # {'waveform': [bs, 1 ch, wlen]}
        target_dict = {'waveform': batch_gt_sel}
        loss_dict = self.loss_func(output_dict, target_dict)

        return batchsize, loss_dict

    def validation_step_processing(self, batch_data_dict, batch_idx):
        """
        process batch_data_dict and return loss and metrics
        one of loss_or_metric_name in loss_dict can be selected for ModelCheckpoint(monitor)

        Returns:
            batchsize (int)
            loss_dict (dict)
                {
                    "loss_or_metric_name": val # for logging, 'loss' is not required
                }
        """
        batchsize = batch_data_dict['mixture'].shape[0]
        nsources = batch_data_dict['dry_sources'].shape[1]

        batch_labels = batch_data_dict['label']
        batch_gt = batch_data_dict['dry_sources'] # [bs, nsources, 1 ch, wlen]
        batch_label_vector = batch_data_dict['label_vector'] # [bs, nsources x nclasses]
        batch_label_vector_stack = batch_label_vector.view(batchsize, nsources, batch_label_vector.shape[-1]//nsources)# [bs, nsources, nclasses]

        selected_idx = self._se_selection(batch_labels)
        batch_gt_sel = torch.stack([batch_gt[i, se_idx, :, :] for i, se_idx in enumerate(selected_idx)], dim=0) # [bs, 1 ch, wlen]
        batch_label_vector_sel = torch.stack([batch_label_vector_stack[i, se_idx, :] for i, se_idx in enumerate(selected_idx)], dim=0) # [bs, nclasses]

        input_dict = {
            'mixture': batch_data_dict['mixture'], # [bs, nch, wlen]
            'label_vector': batch_label_vector_sel # [bs, label_len]
        }
        output_dict = self.model(input_dict) # {'waveform': [bs, 1 ch, wlen]}
        target_dict = {'waveform': batch_gt_sel}
        loss_dict = self.loss_func(output_dict, target_dict)

        loss_dict = {k: v.item() for k,v in loss_dict.items()}
        if self.metric_func: # add metrics
            metric = self.metric_func(output_dict, target_dict)
            for k,v in metric.items():
                loss_dict[k] = v.mean().item() # torch tensor size [bs]

        return batchsize, loss_dict

# if __name__ == '__main__':
#     # cd ../..
#
#     import os, sys; os.chdir('../..'); sys.path.append(os.getcwd())
#     import importlib
#     def initialize_config(module_cfg):
#         module = importlib.import_module(module_cfg["module"])
#         if 'args' in module_cfg.keys(): return getattr(module, module_cfg["main"])(**module_cfg["args"])
#         return getattr(module, module_cfg["main"])()
#
#     config = {
#         'module': 'src.training.label_query_separation',
#         'main': 'LabelQuerySeparationLightning',
#         'args': {
#             'model': {
#                 "module": "src.models.resunet.resunet_mono_out",
#                 "main": "ResUNet30",
#                 "args":{
#                     "input_channels": 2,
#                     "ref_channel": 1,
#                     "label_len": 20,
#                 }
#             },
#             'loss': {
#                 "module": 'src.training.loss.snr_sisnr',
#                 'main': 'get_loss_func',
#                 'args': {
#                     'snr_weight': 0.9,
#                     'sisnr_weight': 0.1
#                 }
#             },
#             'metric': {
#                 "module": 'src.training.metrics.metrics',
#                 'main': 'get_metric_func'
#             },
#             'optimizer': {
#                 'module': 'torch.optim',
#                 'main': 'AdamW',
#                 'args': {
#                     'params': 'assigned in src.training.label_query_separation',
#                     'lr': 0.00001,
#                     'betas': (0.9, 0.999),
#                     'eps': 1e-08,
#                     'weight_decay': 0.0,
#                     'amsgrad': True
#                 }
#             },
#             'is_validation': True
#         }
#     }
#
#     module = initialize_config(config)
    
