from typing import Optional
import math
import json
from pathlib import Path

import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.testing.utils.ldm_utils import load_models, suppress_logs

# configs
out_dir = './results/gen_infill'
batch_size = 16
device = 'cuda'
show_plots = False
force_original_content = False # set to true, if only the mask can be changed
deterministic_x_T = False # this should always be False for some reason...

# model configs
model_configs = {
    'shape_ldm': {
        'exp_name': '2024-10-23T23-10-29_shape-ldm-tiny',
        'epoch': 291},
    'height_ldm': {
        'exp_name': '2024-10-28T22-11-39_height-ldm-small',
        'epoch': 135},
}


with suppress_logs():
    shape_model, height_model = load_models(model_configs)

out_dir = Path(out_dir)
out_dir = out_dir / f'{force_original_content=}_{deterministic_x_T=}'
out_dir.mkdir(exist_ok=True, parents=True)



# load input images
shape_im = np.load('./data/samples/shapes/shape_0.npy')
height_im = np.load('./data/samples/heights/height_0.npy')
# shape_cond_scalar = shape_im.mean()

# plot orig image
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(shape_im)
axs[0].set_title('Shape')
axs[0].axis('off')
axs[1].imshow(height_im)
axs[1].set_title('Height')
axs[1].axis('off')
plt.tight_layout()
plt.savefig(out_dir / 'original.png', dpi=300)
if show_plots:
    plt.show()
else:
    plt.close(fig)


# shape_im = torch.from_numpy(shape_im).to(device)
# height_im = torch.from_numpy(height_im).to(device)
# mask = torch.ones_like(shape_im)
mask = np.ones_like(shape_im)

# mask out the center
mask[
    shape_im.shape[0] // 4:3 * shape_im.shape[0] // 4,
    shape_im.shape[1] // 4:3 * shape_im.shape[1] // 4] = 0.

plt.imshow(mask)
plt.title('Mask, 1s are kept, 0s are filled')
plt.axis('off')
plt.colorbar()
plt.savefig(out_dir / 'mask.png', dpi=300)
if show_plots:
    plt.show()
else:
    plt.close()



def regenerate_shape(
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

    # embed shape
    shape_im = shape_im[:, None, ...] if im_has_batch else shape_im[None, None, ...]
    shape_emb = shape_model.first_stage_model.encode(shape_im).mode()

    # get the conds and masks
    shape_emb = shape_emb.repeat(batch_size, 1, 1, 1) if not im_has_batch else shape_emb
    shape_cond = shape_model.cond_stage_model.sample(batch_size, conds=shape_cond_scalar).to(device)
    mask = mask[None, None, ...]
    mask_emb = torch.nn.functional.interpolate(mask, size=(shape_emb.shape[2], shape_emb.shape[3]), mode='bilinear')
    mask_emb = (~(mask_emb < 1)).float()
    mask_emb = mask_emb.repeat(batch_size, 1, 1, 1) if not im_has_batch else mask_emb
    x_T = torch.rand_like(shape_emb) if deterministic_x_T else None

    z_0_shape, _ = shape_model.sample_log(
        shape_cond,
        batch_size,
        ddim=True,
        ddim_steps=200,
        x0=shape_emb,
        x_T=x_T,
        mask=mask_emb)

    # reconstruct shape
    x_0_shape = shape_model.first_stage_model.decode(z_0_shape)
    x_0_shape = shape_model.first_stage_model.reconstruction_to_image(x_0_shape)
    if force_original_content:
        x_0_shape = shape_im * mask + x_0_shape * (1 - mask)
    
    return x_0_shape.detach().cpu().squeeze(1).numpy()


def regenerate_height(
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

    # embed height
    height_im = height_im[:, None,  ...] if im_has_batch else height_im[None, None, ...]
    height_emb = height_model.first_stage_model.encode(height_im).mode()

    # embed shape
    shape_im = shape_im[:, None, ...] if im_has_batch else shape_im[None, None, ...]
    shape_emb = height_model.cond_stage_model.encode(shape_im).mode()

    # get the conds and masks
    height_emb = height_emb.repeat(batch_size, 1, 1, 1) if not im_has_batch else height_emb
    shape_emb = shape_emb.repeat(batch_size, 1, 1, 1) if not im_has_batch else shape_emb
    x_T = torch.rand_like(height_emb) if deterministic_x_T else None
    mask = mask[None, None, ...]
    mask_emb = torch.nn.functional.interpolate(mask, size=(shape_emb.shape[2], shape_emb.shape[3]), mode='bilinear')
    mask_emb = (~(mask_emb < 1)).float()
    mask_emb = mask_emb.repeat(batch_size, 1, 1, 1)

    z_0_height, _ = height_model.sample_log(
        shape_emb,
        batch_size,
        ddim=True,
        ddim_steps=200,
        x0=height_emb,
        x_T=x_T,
        mask=mask_emb)

    # reconstruct height
    x_0_height = height_model.first_stage_model.decode(z_0_height)
    x_0_height = height_model.first_stage_model.reconstruction_to_image(x_0_height, shape_cond=shape_im.repeat(batch_size, 1, 1, 1) if not im_has_batch else shape_im)
    if force_original_content:
        x_0_height = height_im * mask + x_0_height * (1 - mask)

    return x_0_height.detach().cpu().squeeze(1).numpy()



def regenerate_shape_and_height(
    shape_im_orig: np.ndarray,
    height_im_orig: np.ndarray,
    mask: np.ndarray,
    *,
    batch_size: Optional[int] = 1,
    force_original_content: bool = False,
    deterministic_x_T: bool = False,
    device: str = 'cuda'
) -> np.ndarray:
    """
    shape_im_orig: shape (128, 128)
    height_im_orig: shape (128, 128)
    mask: shape (128, 128) with bool values

    Returns: shape (batch_size, 128, 128)
    """

    assert shape_im_orig.shape == height_im_orig.shape
    im_has_batch = (shape_im_orig.ndim == 3)
    assert im_has_batch != (batch_size is not None), 'Batch size must be provided through either shape_im or batch_size, not both'
    

    shape_x_0 = regenerate_shape(
        shape_im_orig,
        mask,
        batch_size=batch_size,
        force_original_content=force_original_content,
        deterministic_x_T=deterministic_x_T,
        device=device)
    
    if not im_has_batch:
        height_im_orig = np.repeat(height_im_orig[None, ...], batch_size, axis=0)
    
    height_x_0 = regenerate_height(
        height_im_orig,
        shape_x_0,
        mask,
        batch_size=None,
        force_original_content=force_original_content,
        deterministic_x_T=deterministic_x_T,
        device=device)
    
    return shape_x_0, height_x_0
    



# ---------- shape generative infill ----------
x_0_shape = regenerate_shape(
    shape_im,
    mask,
    batch_size=batch_size,
    force_original_content=force_original_content,
    deterministic_x_T=deterministic_x_T, device=device)

# save the generated shapes
fig_size = int(np.sqrt(batch_size))
assert fig_size ** 2 == batch_size, 'Batch size must be a square number'
fig, axs = plt.subplots(fig_size, fig_size, figsize=(fig_size*5, fig_size*5))
for _shape, ax in zip(x_0_shape, axs.flatten()):
    ax.imshow(_shape)
    ax.axis('off')
plt.tight_layout()
plt.savefig(out_dir / 'generated_shapes.png', dpi=300)
if show_plots:
    plt.show()
else:
    plt.close(fig)
    

# ---------- height generative infill ----------
x_0_height = regenerate_height(
    height_im,
    shape_im,
    mask,
    batch_size=batch_size,
    force_original_content=force_original_content,
    deterministic_x_T=deterministic_x_T,
    device=device)


# save the generated heights
fig, axs = plt.subplots(fig_size, fig_size, figsize=(fig_size*5, fig_size*5))
for _height, ax in zip(x_0_height, axs.flatten()):
    ax.imshow(_height)
    ax.axis('off')
plt.tight_layout()
plt.savefig(out_dir / 'generated_heights.png', dpi=300)
if show_plots:
    plt.show()
else:
    plt.close(fig)


# ---------- shape and height generative infill ----------
x_0_shape, x_0_height = regenerate_shape_and_height(
    shape_im,
    height_im,
    mask,
    batch_size=batch_size,
    force_original_content=force_original_content,
    deterministic_x_T=deterministic_x_T,
    device=device)

# save the generated shapes and heights
fig, axs = plt.subplots(fig_size, 2*fig_size, figsize=(fig_size*10, fig_size*5))
shape_axs = axs[:, :fig_size]
height_axs = axs[:, fig_size:]

for _shape, _height, shape_ax, height_ax in zip(x_0_shape, x_0_height, shape_axs.flatten(), height_axs.flatten()):
    shape_ax.imshow(_shape)
    shape_ax.axis('off')
    height_ax.imshow(_height)
    height_ax.axis('off')
plt.tight_layout()
plt.savefig(out_dir / 'generated_shapes_and_heights.png', dpi=300)
if show_plots:
    plt.show()
else:
    plt.close(fig)
