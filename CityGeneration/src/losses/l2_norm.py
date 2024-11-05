



from torch import nn
import torch

class L2Norm(nn.Module):
    def __init__(self, exp_space = False, reduce = False, **kwargs):
        """
       
        """
        super(L2Norm, self).__init__()
        self.loss_func = nn.MSELoss(reduce = reduce)
        self.exp_space = exp_space

    def forward(self, y_pred, y_true):
        
        if self.exp_space:
            return torch.sum(self.loss_func(torch.exp(y_pred), torch.exp(y_true)))
        return torch.sum(self.loss_func(y_pred, y_true))
    
    