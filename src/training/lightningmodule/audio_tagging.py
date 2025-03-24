from .base_lightningmodule import BaseLightningModule

class AudioTagging(BaseLightningModule):
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

        input_dict = {
            'waveform': batch_data_dict['mixture'], # [bs, wlen]
        }
        output_dict = self.model(input_dict) # {'probabilities': [bs, nclasses]}
        target_dict = {'probabilities': batch_data_dict['label_vector']}
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

        input_dict = {
            'waveform': batch_data_dict['mixture'], # [bs, wlen]
        }
        output_dict = self.model(input_dict) # {'probabilities': [bs, nclasses]}
        target_dict = {'probabilities': batch_data_dict['label_vector']}
        loss_dict = self.loss_func(output_dict, target_dict)

        loss_dict = {k: v.item() for k,v in loss_dict.items()}
        if self.metric_func: # add metrics
            metric = self.metric_func(output_dict, target_dict)
            for k,v in metric.items(): # metric return [bs] for better calculation of mean
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
    
