

from torch import nn
import numpy as np
import torch

from CityGeneration.src.utils.image_utils import crop_to_square, resize
from torchvision import transforms as T

class InferenceModel(nn.Module):
    def __init__(self, model : nn.Module, img_size, **kwargs):
        super().__init__()
        self.model = model
        self.img_size = img_size

    def forward(self, input : np.ndarray):
        prepared = self._prepare(input)
        return torch.exp(self.model(prepared))
    
    def _prepare(self, input : np.ndarray) -> np.ndarray:
        input = crop_to_square(input)
        input = resize(input, (self.img_size, self.img_size))
        
        input = input / 500 # normalization
        input = input.astype(np.float32)
    
        input = T.ToTensor()(input)[None,...]
    
        return input
        
        