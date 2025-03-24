from typing import Any, Callable, Dict
import lightning.pytorch as pl
import torch
from huggingface_hub import PyTorchModelHubMixin
import importlib

from src.utils import initialize_config

class BaseLightningModule(pl.LightningModule, PyTorchModelHubMixin):
    def __init__(
        self,
        model: Dict,
        loss: Dict,
        optimizer: Dict,
        lr_scheduler:Dict=None,
        is_validation=False,
        metric:Dict=None,
    ):
        """
        Module for training the label queried audio separation
        All the dict input are configs for initializing the modules
        This is an abstract class
        training_step_processing and validation_step_processing must be implemeted in the subclass
        
        Args:
            model (dict)
            loss (dict)
            optimizer (dict)
            lr_scheduler (dict, optional)
            is_validation (dict, optional)
            metric (dict, optional): if not provided, only calculate loss on for validation
        """
        super().__init__()
        self.model_config = model
        self.model = initialize_config(self.model_config)

        self.loss_config = loss
        self.loss_func = initialize_config(self.loss_config)

        self.optimizer_config = optimizer
        self.optimizer_config['args']['params'] = self.model.parameters() # modify if some parts are frozen
        self.optimizer = initialize_config(self.optimizer_config)

        self.lr_scheduler_config = lr_scheduler
        if self.lr_scheduler_config: # can be optional
            self.lr_scheduler_config['scheduler']['args']['optimizer'] = self.optimizer
            # if scheduler is LambdaLR, initialize the lambda function
            if self.lr_scheduler_config['scheduler']['main'] == 'LambdaLR':
                self.lr_lamda_config = self.lr_scheduler_config['scheduler']['args']['lr_lambda']
                self.lr_lambda = initialize_config(self.lr_lamda_config)
                self.lr_scheduler_config['scheduler']['args']['lr_lambda'] = self.lr_lambda
            self.scheduler = initialize_config(self.lr_scheduler_config['scheduler'])

        if is_validation:
            self.validation_step = self._validation_step
            if metric:
                self.metric_config = metric
                self.metric_func = initialize_config(self.metric_config)
            else:
                self.metric_func = None
        
        self.is_validation = is_validation

    def forward(self, x):
        pass
    
    def set_train_mode(self):
        self.model.train()

    def training_step_processing(self, batch_data_dict, batch_idx):
        raise NotImplementedError
        """
        process batch_data_dict and return loss

        Args:
            batch_data_dict (dict)
            batch_idx
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
            'mixture': batch_data_dict['mixture'], # [bs, nch, wlen]
            'label_vector': batch_data_dict['label_vector'] # [bs, label_len]
            }
        output_dict = self.model(input_dict) # {'waveform': [bs, nch, wlen]}
        target_dict = {'waveform': batch_data_dict['ground_truth']}
        loss_dict = self.loss_func(output_dict, target_dict)

        return batchsize, loss_dict

    def training_step(self, batch_data_dict, batch_idx):
        """
        Args:
            batch_data_dict: a mini batch from dataloader
            batch_idx: int

        Returns:
            loss: float, loss function of this mini-batch
        """
        self.set_train_mode()

        batchsize, loss_dict = self.training_step_processing(batch_data_dict, batch_idx)

        loss = loss_dict['loss'] # for back propagation

        # log all items in loss_dict
        step_dict = {f'step_train/{name}': val.item() for name, val in loss_dict.items()}
        self.log_dict(step_dict, prog_bar=False, logger=True, on_epoch=False, on_step=True, sync_dist=True, batch_size=batchsize)
        epoc_dict = {f'epoch_train/{name}': val.item() for name, val in loss_dict.items()}
        self.log_dict(epoc_dict, prog_bar=True, logger=True, on_epoch=True, on_step=False, sync_dist=True, batch_size=batchsize)
        
        self.log_dict({"epoch/lr": self.optimizer.param_groups[0]['lr']},)

        return loss


    def validation_step_processing(self, batch_data_dict, batch_idx):
        raise NotImplementedError
        """
        process batch_data_dict and return loss and metrics
        one of loss_or_metric_name in loss_dict can be selected for ModelCheckpoint(monitor)

        Args:
            batch_data_dict (dict)
            batch_idx
        Returns:
            batchsize (int)
            loss_dict (dict)
                {
                    "loss_or_metric_name": val # for logging, 'loss' is not required
                }
        """
        batchsize = batch_data_dict['mixture'].shape[0]

        input_dict = {
            'mixture': batch_data_dict['mixture'], # [bs, nch, wlen]
            'label_vector': batch_data_dict['label_vector'] # [bs, label_len]
            }
        output_dict = self.model(input_dict) # {'waveform': [bs, nch, wlen]}
        target_dict = {'waveform': batch_data_dict['ground_truth']}
        loss_dict = self.loss_func(output_dict, target_dict)

        loss_dict = {k: v.item() for k,v in loss_dict.items()}
        if self.metric_func: # add metrics
            metric = self.metric_func(output_dict, target_dict)
            for k,v in metric.items():
                loss_dict[k] = v.mean().item() # torch tensor size [bs]

        return batchsize, loss_dict

    def _validation_step(self, batch_data_dict, batch_idx):
        """
        Args:
            batch_data_dict: a mini batch from dataloader
            batch_idx: int

        Returns:
            None
        """
        self.model.eval()

        batchsize, loss_dict = self.validation_step_processing(batch_data_dict, batch_idx)

        # log all items in loss_dict
        step_dict = {f'step_val/{name}': metric for name, metric in loss_dict.items()}
        self.log_dict(step_dict, prog_bar=False, logger=True, on_epoch=False, on_step=True, sync_dist=True, batch_size=batchsize)
        epoc_dict = {f'epoch_val/{name}': metric for name, metric in loss_dict.items()}
        self.log_dict(epoc_dict, prog_bar=True, logger=True, on_epoch=True, on_step=False, sync_dist=True, batch_size=batchsize)

    def configure_optimizers(self):
        r"""Configure optimizer.
            will be called automatically
        """
        if self.lr_scheduler_config:
            return {
                "optimizer": self.optimizer,
                "lr_scheduler": {
                    'scheduler': self.scheduler,
                    'interval': self.lr_scheduler_config['interval'],
                    'frequency': self.lr_scheduler_config['frequency'],
                }
            }
        else:
            return self.optimizer
    
if __name__ == '__main__':
    # cd ../..

    import os, sys; os.chdir('../..'); sys.path.append(os.getcwd())
    import importlib
    def initialize_config(module_cfg):
        module = importlib.import_module(module_cfg["module"])
        if 'args' in module_cfg.keys(): return getattr(module, module_cfg["main"])(**module_cfg["args"])
        return getattr(module, module_cfg["main"])()

    config = {
        'module': 'src.training.label_query_separation',
        'main': 'LabelQuerySeparationLightning',
        'args': {
            'model': {
                "module": "src.models.resunet.resunet_mono_out",
                "main": "ResUNet30",
                "args":{
                    "input_channels": 2,
                    "ref_channel": 1,
                    "label_len": 20,
                }
            },
            'loss': {
                "module": 'src.training.loss.snr_sisnr',
                'main': 'get_loss_func',
                'args': {
                    'snr_weight': 0.9,
                    'sisnr_weight': 0.1
                }
            },
            'metric': {
                "module": 'src.training.metrics.metrics',
                'main': 'get_metric_func'
            },
            'optimizer': {
                'module': 'torch.optim',
                'main': 'AdamW',
                'args': {
                    'params': 'assigned in src.training.label_query_separation',
                    'lr': 0.00001,
                    'betas': (0.9, 0.999),
                    'eps': 1e-08,
                    'weight_decay': 0.0,
                    'amsgrad': True
                }
            },
            'is_validation': True
        }
    }

    module = initialize_config(config)
    
