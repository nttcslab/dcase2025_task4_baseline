from .base_lightningmodule import BaseLightningModule

class AudioTagging(BaseLightningModule):
    def training_step_processing(self, batch_data_dict, batch_idx):
        batchsize = batch_data_dict['mixture'].shape[0]

        input_dict = {
            'waveform': batch_data_dict['mixture'], # [bs, wlen]
        }
        output_dict = self.model(input_dict) # {'probabilities': [bs, nclasses]}
        target_dict = {'probabilities': batch_data_dict['label_vector']}
        loss_dict = self.loss_func(output_dict, target_dict)

        return batchsize, loss_dict

    def validation_step_processing(self, batch_data_dict, batch_idx):
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


