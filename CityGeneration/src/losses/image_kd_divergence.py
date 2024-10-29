

from torch import nn

class ImageKLDivergence(nn.Module):
    def __init__(self, reduction = "mean", log_target = False, **kwargs):
        """
        KL divergence loss for image data of shape (B, C, H, W).
        See: https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html for more infor.
        
        Args:
            reduction (str): Reduction method: mean (default), batchmean, sum
            log_target (bool): Whether the target is in log space (default: False)
        """
        super(ImageKLDivergence, self).__init__()
        
        self.loss_func = nn.KLDivLoss(reduction = reduction, log_target = log_target)

    def forward(self, y_pred, y_true):
        
        return self.loss_func(y_pred, y_true)