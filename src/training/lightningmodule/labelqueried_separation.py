from .base_lightningmodule import BaseLightningModule

class LabelQueriedSeparationLightning(BaseLightningModule):
    def training_step_processing(self, batch_data_dict, batch_idx):
        batchsize = batch_data_dict['mixture'].shape[0]

        input_dict = {
            'mixture': batch_data_dict['mixture'], # [bs, nch, wlen]
            'label_vector': batch_data_dict['label_vector'] # [bs, label_len]
        }
        output_dict = self.model(input_dict) # {'waveform': [bs, nch, wlen]}
        target_dict = {'waveform': batch_data_dict['dry_sources']}
        loss_dict = self.loss_func(output_dict, target_dict)

        return batchsize, loss_dict

    def validation_step_processing(self, batch_data_dict, batch_idx):
        batchsize = batch_data_dict['mixture'].shape[0]

        input_dict = {
            'mixture': batch_data_dict['mixture'], # [bs, nch, wlen]
            'label_vector': batch_data_dict['label_vector'] # [bs, label_len]
        }
        output_dict = self.model(input_dict) # {'waveform': [bs, nch, wlen]}
        target_dict = {'waveform': batch_data_dict['dry_sources']}
        loss_dict = self.loss_func(output_dict, target_dict)

        loss_dict = {k: v.item() for k,v in loss_dict.items()}
        if self.metric_func: # add metrics
            metric = self.metric_func(output_dict, target_dict)
            for k,v in metric.items():
                loss_dict[k] = v.mean().item() # torch tensor size [bs]

        return batchsize, loss_dict
