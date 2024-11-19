import math
import json
from pathlib import Path

import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.testing.utils.ldm_utils import load_models, suppress_logs
     
class HeightShapeGenerator:
    def __init__(self, model_configs = None, device='cuda'):
        if model_configs is None:
            model_configs = {
                'shape_ldm': {
                    'exp_name': '2024-10-23T23-10-29_shape-ldm-tiny',
                    'epoch': 291},
                'height_ldm': {
                    'exp_name': '2024-10-28T22-11-39_height-ldm-small',
                    'epoch': 135},
            }
            
        with suppress_logs():
            self.shape_model, self.height_model = load_models(model_configs)
            
        model_configs['configs'] = dict()
        # model_configs['configs']['exp_name'] = exp_name
        model_configs['configs']['batch_size'] = 1
        model_configs['configs']['num_samples'] = 1
        model_configs['configs']['device'] = device
        
        self.device = device
        self.configs = model_configs
        
    def generate_shape(self, size_cond = 0.5, ddims_steps = 200, batch_size=1) -> np.ndarray:
        with torch.no_grad():
            with suppress_logs():
                cond, target_areas = self.shape_model.cond_stage_model.sample(batch_size, return_scalers=True, conds=size_cond)
                z_0_shape, _ = self.shape_model.sample_log(cond.to(self.device), batch_size, ddim=True, ddim_steps=ddims_steps)

                x_shape_logits = self.shape_model.decode_first_stage(z_0_shape)
                shape_im = self.shape_model.first_stage_model.reconstruction_to_image(x_shape_logits)
        return shape_im.squeeze(1).squeeze(0).cpu().numpy()
                 
    def generate_height(self, shape: np.ndarray, ddim_steps = 200, batch_size=1) -> np.ndarray:
        shape = torch.tensor(shape).to(self.device)
        shape = shape.unsqueeze(0).unsqueeze(1) 
        
        with torch.no_grad():
            with suppress_logs():
                s_dist = self.shape_model.encode_first_stage(shape)
                z_0_shape = s_dist.sample()[0].unsqueeze(0)
                z_0_height, _ = self.height_model.sample_log(z_0_shape, batch_size, ddim=True, ddim_steps=ddim_steps)

                x_height_logits = self.height_model.decode_first_stage(z_0_height)
                height_im = self.height_model.first_stage_model.reconstruction_to_image(x_height_logits, shape_cond=shape)
        
        return height_im.squeeze(1).squeeze(0).cpu().numpy()