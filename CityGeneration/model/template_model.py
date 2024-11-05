

from torch import nn
import torch

class TemplateModel(nn.Module):
    
    def __init__(self, param_a, param_b, **kwargs):
        super().__init__()
        
        self.param_a = param_a
        self.param_b = param_b
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return x