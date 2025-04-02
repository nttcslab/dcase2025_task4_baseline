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
