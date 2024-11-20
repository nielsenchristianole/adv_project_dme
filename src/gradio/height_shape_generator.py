
from typing import Optional, Tuple

import numpy as np
import torch

from src.testing.utils.ldm_utils import load_models, suppress_logs
# from src.testing.generate_infill import regenerate_shape, regenerate_height, regenerate_shape_and_height

     
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
        
    def generate_shape(self, size_cond = None, ddims_steps = 200, batch_size=1) -> np.ndarray:
        with torch.no_grad():
            with suppress_logs():
                cond, target_areas = self.shape_model.cond_stage_model.sample(batch_size, return_scalers=True, conds=size_cond)
                z_0_shape, _ = self.shape_model.sample_log(cond.to(self.device), batch_size, ddim=True, ddim_steps=ddims_steps)

                x_shape_logits = self.shape_model.decode_first_stage(z_0_shape)
                shape_im = self.shape_model.first_stage_model.reconstruction_to_image(x_shape_logits)
        return shape_im.detach().cpu().squeeze(1).numpy()
                 
    def generate_height(self, shape: np.ndarray, ddim_steps = 200, batch_size=1) -> np.ndarray:
        shape = torch.tensor(shape).to(self.device)
        shape = shape.unsqueeze(0).unsqueeze(1) 
        
        with torch.no_grad():
            with suppress_logs():
                s_dist = self.shape_model.encode_first_stage(shape)
                z_0_shape = s_dist.mode()[0].unsqueeze(0)
                z_0_height, _ = self.height_model.sample_log(z_0_shape, batch_size, ddim=True, ddim_steps=ddim_steps)

                x_height_logits = self.height_model.decode_first_stage(z_0_height)
                height_im = self.height_model.first_stage_model.reconstruction_to_image(x_height_logits, shape_cond=shape)
        
        return height_im.detach().cpu().squeeze(1).numpy()
    
    def regenerate_shape(
            self,
            shape_im_orig: np.ndarray,
            mask: np.ndarray,
            *,
            batch_size: Optional[int] = 1,
            force_original_content: bool = False,
            deterministic_x_T: bool = False,
            device='cuda'
        ) -> np.ndarray:
        """
        shape_im_orig: shape (128, 128)
        mask: shape (128, 128) with bool values

        Returns: shape (batch_size, 128, 128)
        """
        assert shape_im_orig.shape[-2:] == (128, 128), 'Shape must be 128x128'
        im_has_batch = (shape_im_orig.ndim == 3)
        assert im_has_batch != (batch_size is not None), 'Batch size must be provided through either shape_im or batch_size, not both'
        assert mask.shape == shape_im_orig.shape[-2:] and mask.ndim == 2, 'Mask must be 128x128'
        batch_size = batch_size or shape_im_orig.shape[0]

        mask = (mask > 0).copy()
        shape_im = (shape_im_orig.copy() > 0)
        shape_cond_scalar = mask.mean(axis=(1, 2)) if im_has_batch else mask.mean()

        shape_im = torch.from_numpy(shape_im.astype(np.float32)).to(device)
        mask = torch.from_numpy(mask.astype(np.float32)).to(device)
        
        with torch.no_grad():
            with suppress_logs():
                # embed shape
                shape_im = shape_im[:, None, ...] if im_has_batch else shape_im[None, None, ...]
                shape_emb = self.shape_model.first_stage_model.encode(shape_im).mode()

                # get the conds and masks
                shape_emb = shape_emb.repeat(batch_size, 1, 1, 1) if not im_has_batch else shape_emb
                shape_cond = self.shape_model.cond_stage_model.sample(batch_size, conds=shape_cond_scalar).to(device)
                mask = mask[None, None, ...]
                mask_emb = torch.nn.functional.interpolate(mask, size=(shape_emb.shape[2], shape_emb.shape[3]), mode='bilinear')
                mask_emb = (~(mask_emb < 1)).float()
                mask_emb = mask_emb.repeat(batch_size, 1, 1, 1) if not im_has_batch else mask_emb
                x_T = torch.rand_like(shape_emb) if deterministic_x_T else None

                z_0_shape, _ = self.shape_model.sample_log(
                    shape_cond,
                    batch_size,
                    ddim=True,
                    ddim_steps=200,
                    x0=shape_emb,
                    x_T=x_T,
                    mask=mask_emb)

                # reconstruct shape
                x_0_shape = self.shape_model.first_stage_model.decode(z_0_shape)
                x_0_shape = self.shape_model.first_stage_model.reconstruction_to_image(x_0_shape)
                if force_original_content:
                    x_0_shape = shape_im * mask + x_0_shape * (1 - mask)
        
        return x_0_shape.detach().cpu().squeeze(1).numpy()


    def regenerate_height(
        self,
        height_im_orig: np.ndarray,
        shape_im_orig: Optional[np.ndarray],
        mask: np.ndarray,
        *,
        batch_size: int = 1,
        device: str = 'cuda',
        deterministic_x_T: bool = False,
        force_original_content: bool = False
    ) -> np.ndarray:
        """
        shape_im_orig: shape (128, 128) or None (the generated from height)
        height_im_orig: shape (128, 128)
        mask: shape (128, 128) with bool values
        """

        assert height_im_orig.shape[-2:] == (128, 128), 'Height must be 128x128'
        assert shape_im_orig is None or shape_im_orig.shape[-2:] == (128, 128), 'Shape must be 128x128'
        assert mask.shape == shape_im_orig.shape[-2:] and mask.ndim == 2, 'Mask must be 128x128'

        assert ((height_im_orig.ndim == shape_im_orig.ndim == 3) and batch_size is None) or \
            ((height_im_orig.ndim == shape_im_orig.ndim == 2) and batch_size is not None), 'Batch size must be provided through either height_im or batch_size, not both'
        im_has_batch = (height_im_orig.ndim == 3)
        batch_size = batch_size or height_im_orig.shape[0]

        shape_im = (shape_im_orig.copy() > 0) if shape_im_orig is not None else (height_im_orig > 0)
        height_im = height_im_orig.copy()

        mask = (mask > 0).copy()

        shape_im = torch.from_numpy(shape_im.astype(np.float32)).to(device)
        height_im = torch.from_numpy(height_im.astype(np.float32)).to(device)
        mask = torch.from_numpy(mask.astype(np.float32)).to(device)

        with torch.no_grad():
            with suppress_logs():
                # embed height
                height_im = height_im[:, None,  ...] if im_has_batch else height_im[None, None, ...]
                height_emb = self.height_model.first_stage_model.encode(height_im).mode()

                # embed shape
                shape_im = shape_im[:, None, ...] if im_has_batch else shape_im[None, None, ...]
                shape_emb = self.height_model.cond_stage_model.encode(shape_im).mode()

                # get the conds and masks
                height_emb = height_emb.repeat(batch_size, 1, 1, 1) if not im_has_batch else height_emb
                shape_emb = shape_emb.repeat(batch_size, 1, 1, 1) if not im_has_batch else shape_emb
                x_T = torch.rand_like(height_emb) if deterministic_x_T else None
                mask = mask[None, None, ...]
                mask_emb = torch.nn.functional.interpolate(mask, size=(shape_emb.shape[2], shape_emb.shape[3]), mode='bilinear')
                mask_emb = (~(mask_emb < 1)).float()
                mask_emb = mask_emb.repeat(batch_size, 1, 1, 1)

                z_0_height, _ = self.height_model.sample_log(
                    shape_emb,
                    batch_size,
                    ddim=True,
                    ddim_steps=200,
                    x0=height_emb,
                    x_T=x_T,
                    mask=mask_emb)

                # reconstruct height
                x_0_height = self.height_model.first_stage_model.decode(z_0_height)
                x_0_height = self.height_model.first_stage_model.reconstruction_to_image(x_0_height, shape_cond=shape_im.repeat(batch_size, 1, 1, 1) if not im_has_batch else shape_im)
                if force_original_content:
                    x_0_height = height_im * mask + x_0_height * (1 - mask)

        return x_0_height.detach().cpu().squeeze(1).numpy()

    def regenerate_shape_and_height(
        self,
        shape_im_orig: np.ndarray,
        height_im_orig: np.ndarray,
        mask: np.ndarray,
        *,
        batch_size: Optional[int] = 1,
        force_original_content: bool = False,
        deterministic_x_T: bool = False,
        device: str = 'cuda'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        shape_im_orig: shape (128, 128)
        height_im_orig: shape (128, 128)
        mask: shape (128, 128) with bool values

        Returns: shape (batch_size, 128, 128)
        """

        assert shape_im_orig.shape == height_im_orig.shape
        im_has_batch = (shape_im_orig.ndim == 3)
        assert im_has_batch != (batch_size is not None), 'Batch size must be provided through either shape_im or batch_size, not both'
        

        shape_x_0 = self.regenerate_shape(
            shape_im_orig,
            mask,
            batch_size=batch_size,
            force_original_content=force_original_content,
            deterministic_x_T=deterministic_x_T,
            device=device)
        
        if not im_has_batch:
            height_im_orig = np.repeat(height_im_orig[None, ...], batch_size, axis=0)
        
        height_x_0 = self.regenerate_height(
            height_im_orig,
            shape_x_0,
            mask,
            batch_size=None,
            force_original_content=force_original_content,
            deterministic_x_T=deterministic_x_T,
            device=device)
        
        return shape_x_0, height_x_0