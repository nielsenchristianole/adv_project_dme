import math
import json
from pathlib import Path

import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.testing.utils.ldm_utils import load_models, suppress_logs

# configs
out_dir = './data/samples/'
batch_size = 16
device = 'cuda'

# uncond samples
num_samples = 1000

# cond samples
num_samples_per_cond = 200
conds = np.arange(0, 0.75 + 0.05, 0.05)

# dataset samples
sample_train = False
sample_test = False

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
out_dir.mkdir(exist_ok=True, parents=True)

# exp_name = f'exp_{len(list(out_dir.iterdir()))}'
# out_dir /= exp_name
out_dir.mkdir(exist_ok=True)

(out_dir / 'shapes').mkdir(exist_ok=True)
(out_dir / 'heights').mkdir(exist_ok=True)
(out_dir / 'shape_conds').mkdir(exist_ok=True)

# save configs
model_configs['configs'] = dict()
# model_configs['configs']['exp_name'] = exp_name
model_configs['configs']['batch_size'] = batch_size
model_configs['configs']['num_samples'] = num_samples
model_configs['configs']['device'] = device
with open(out_dir / 'config.json', 'w') as f:
    json.dump(model_configs, f, indent=2)


# get uncond shapes
with torch.no_grad():

    pbar = tqdm.trange(num_samples, desc='Generating uncond samples')

    _sample_counter = len(list((out_dir / 'shapes').iterdir()))
    while pbar.n < num_samples:

        _batch_size = min(batch_size, num_samples - pbar.n)

        # shape
        with suppress_logs():
            cond, target_areas = shape_model.cond_stage_model.sample(_batch_size, return_scalers=True)
            z_0_shape, _ = shape_model.sample_log(cond.to(device), _batch_size, ddim=True, ddim_steps=200)

            x_shape_logits = shape_model.decode_first_stage(z_0_shape)
            shape_im = shape_model.first_stage_model.reconstruction_to_image(x_shape_logits)

            # height
            z_0_height, _ = height_model.sample_log(z_0_shape, _batch_size, ddim=True, ddim_steps=200)

            x_height_logits = height_model.decode_first_stage(z_0_height)
            height_im = height_model.first_stage_model.reconstruction_to_image(x_height_logits, shape_cond=shape_im)

        for _shape, _height in zip(shape_im.cpu().numpy().squeeze(1), height_im.cpu().numpy().squeeze(1)):
            np.save(out_dir / 'shapes' / f'shape_{pbar.n + _sample_counter}.npy', _shape)
            np.save(out_dir / 'heights' / f'height_{pbar.n + _sample_counter}.npy', _height)
            pbar.update(1)

            if pbar.n >= num_samples:
                break

    pbar.close()


# get cond shapes
with torch.no_grad():

    for cond_float in tqdm.tqdm(conds, desc='Generating cond samples'):

        _dir = out_dir / 'shape_conds' / f'{cond_float:.2f}'
        _dir.mkdir(exist_ok=True)
        _sample_counter = len(list(_dir.iterdir()))
        for n_missing in tqdm.trange(0, num_samples_per_cond, batch_size, leave=False):

            _batch_size = min(batch_size, num_samples_per_cond - n_missing)

            # shape
            with suppress_logs():
                cond, target_areas = shape_model.cond_stage_model.sample(_batch_size, return_scalers=True, conds=cond_float)
                z_0_shape, _ = shape_model.sample_log(cond.to(device), _batch_size, ddim=True, ddim_steps=200)

                x_shape_logits = shape_model.decode_first_stage(z_0_shape)
                shape_im = shape_model.first_stage_model.reconstruction_to_image(x_shape_logits)

            for _shape in shape_im.cpu().numpy().squeeze(1):
                np.save(_dir / f'shape_{_sample_counter}.npy', _shape)
                _sample_counter += 1



# samples from training set
from shape_dataset import ShapeData
from height_dataset import HeightData

seed = 42069

if sample_train:
    shape_data = ShapeData('./data/contours/df_128/df.shp', im_size=128, mode='train')
    height_data = HeightData('./data/height_contours/df_128/df.shp', im_size=128, mode='train')

    num_data = min(len(shape_data), len(height_data))
    idxs = np.random.default_rng(seed).choice(num_data, num_samples, replace=False)

    _dir = out_dir / 'train'
    _dir.mkdir(exist_ok=True)
    (_dir / 'shapes').mkdir(exist_ok=True)
    (_dir / 'heights').mkdir(exist_ok=True)

    for idx in tqdm.tqdm(idxs, 'training samples'):
        shape = shape_data[idx]['image'].numpy().squeeze(-1)
        height = height_data[idx]['image'].numpy().squeeze(-1)

        np.save(_dir / 'shapes' / f'shape_{idx}.npy', shape)
        np.save(_dir / 'heights' / f'height_{idx}.npy', height)


# samples from test set
if sample_test:
    shape_data = ShapeData('./data/contours/df_128/df.shp', im_size=128, mode='test', augment=False)
    height_data = HeightData('./data/height_contours/df_128/df.shp', im_size=128, mode='test')

    _dir = out_dir / 'test'
    _dir.mkdir(exist_ok=True)
    (_dir / 'shapes').mkdir(exist_ok=True)
    (_dir / 'heights').mkdir(exist_ok=True)

    for idx in tqdm.trange(len(shape_data), desc='test shape samples'):
        shape = shape_data[idx]['image'].numpy().squeeze(-1)
        np.save(_dir / 'shapes' / f'shape_test_{idx}.npy', shape)


    for idx in tqdm.trange(len(height_data), desc='test height samples'):
        height = height_data.__getitem__(idx, mirror=False, angle=0.)['image'].numpy().squeeze(-1)
        np.save(_dir / 'heights' / f'height_test_{idx}.npy', height)


quit()

# plot
cells = math.ceil(math.sqrt(batch_size))
fig, axs = plt.subplots(cells, 2*cells, figsize=(2 * cells * 5, cells * 5))
axs = axs.flatten()

# show shapes
for ax, im, target_area in zip(axs[:batch_size], shape_im.cpu().numpy().squeeze(1), target_areas):
    ax.imshow(im, cmap='gray')
    ax.axis('off')
    ax.set_title(f'Target area: {target_area:.4f}\narea: {im.mean():.4f}')

# show heights
for ax, im in zip(axs[batch_size:], height_im.cpu().numpy().squeeze(1)):
    ax.imshow(im, cmap='gray')
    ax.axis('off')
    ax.set_title(f'Max height: {im.max():.2f}\nMean above sea: {im[im > 0].mean():.2f}')
plt.show()
