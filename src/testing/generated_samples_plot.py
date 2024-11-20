from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sample_dir = './data/samples'
out_dir = './results/shapes_and_heights'

figsize = 6
figlenght = 2
cmap = 'gray'

shapes = list()
for i in range(figsize ** 2):
    shapes.append(np.load(Path(sample_dir) / 'shapes' / f'shape_{i}.npy'))

heights = list()
for i in range(figsize ** 2):
    heights.append(np.load(Path(sample_dir) / 'heights' / f'height_{i}.npy'))


fig, axs = plt.subplots(figsize=(figsize*figlenght, figsize*figlenght), nrows=figsize, ncols=figsize)
for im, ax in zip(shapes, axs.flatten()):
    ax.imshow(im, cmap=cmap)
    ax.axis('off')
plt.suptitle('Uncodintional shape samples')
fig.tight_layout()
plt.savefig(Path(out_dir) / 'shape_uncond_samples.pdf')
plt.close(fig)

fig, axs = plt.subplots(figsize=(figsize*figlenght, figsize*figlenght), nrows=figsize, ncols=figsize)
for im, ax in zip(heights, axs.flatten()):
    ax.imshow(im, cmap=cmap)
    ax.axis('off')
plt.suptitle('Uncodintional height samples')
fig.tight_layout()
plt.savefig(Path(out_dir) / 'height_uncond_samples.pdf')
plt.close(fig)




from height_dataset import HeightData
height_data = HeightData('./data/height_contours/df_128/df.shp', im_size=128, mode='train')

generator = np.random.default_rng(42)
idxs = generator.choice(len(height_data), figsize ** 2, replace=False)

fig, axs = plt.subplots(figsize=(figsize*figlenght, figsize*figlenght), nrows=figsize, ncols=figsize)
for idx, ax in zip(idxs, axs.flatten()):
    shape = height_data.__getitem__(idx, mirror=False, angle=0.)['shape'].numpy().squeeze(-1)
    ax.imshow(shape, cmap=cmap)
    ax.axis('off')
plt.suptitle('Training shape samples')
fig.tight_layout()
plt.savefig(Path(out_dir) / 'shape_train_samples.pdf')
plt.close(fig)

fig, axs = plt.subplots(figsize=(figsize*figlenght, figsize*figlenght), nrows=figsize, ncols=figsize)
for idx, ax in zip(idxs, axs.flatten()):
    height = height_data.__getitem__(idx, mirror=False, angle=0.)['image'].numpy().squeeze(-1)
    ax.imshow(height, cmap=cmap)
    ax.axis('off')
plt.suptitle('Training height samples')
fig.tight_layout()
plt.savefig(Path(out_dir) / 'height_train_samples.pdf')





# _dir = Path(out_dir) / 'train'