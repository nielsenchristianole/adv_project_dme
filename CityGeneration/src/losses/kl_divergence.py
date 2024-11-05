

from torch import nn
import torch

class KLDivergence(nn.Module):
    def __init__(self, reduction = "batchmean", log_target = True, **kwargs):
        """
        KL divergence loss for image data of shape (B, C, H, W).
        See: https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html for more infor.
        
        Args:
            reduction (str): Reduction method: mean (default), batchmean, sum
            log_target (bool): Whether the target is in log space (default: False)
        """
        super(KLDivergence, self).__init__()
        
        self.loss_func = nn.KLDivLoss(reduction = reduction, log_target = log_target)

    def forward(self, y_pred, y_true):
        
        return self.loss_func(y_pred, y_true)