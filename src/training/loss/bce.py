import torch.nn.functional as F
def get_loss_func():
    def loss_func(output, target):
        target_prob = target['probabilities']
        output_prob = output['probabilities']
        loss_val = F.binary_cross_entropy(output_prob, target_prob, reduction='mean')

        loss_dict = {
            'loss': loss_val,
        }
        return loss_dict
    return loss_func