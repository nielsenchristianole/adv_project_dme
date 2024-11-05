
import torch

class PlaceholderLoss(torch.nn.Module):
    def __init__(self, **kwargs):
        super(PlaceholderLoss, self).__init__()

    def forward(self, y_pred, y_true):
        return torch.mean(y_pred - y_true)