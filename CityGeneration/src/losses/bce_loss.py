



from torch import nn
import torch

class BCELoss(nn.Module):
    def __init__(self, reduction = "mean", **kwargs):
        """
       
        """
        super(BCELoss, self).__init__()
        self.loss_func = nn.BCELoss(reduction=reduction)

    def forward(self, y_pred, y_true):
        
        return self.loss_func(y_pred, y_true)
    
    