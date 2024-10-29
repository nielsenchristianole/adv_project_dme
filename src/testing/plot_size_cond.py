import tqdm
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from pathlib import Path
from shape_dataset import ShapeData

from src.testing.utils.distance_utils import load_data


out_dir = './results/shapes_and_heights'
data_dir = './data/samples/shape_conds'
sample_interval = 0.95
data_interval = 1.
im_size = 128
fig_kwargs = dict(
    dpi=300)
show_plots = False
out_ext = 'pdf'


out_dir = Path(out_dir)
out_dir.mkdir(exist_ok=True, parents=True)
path = Path(f'./data/contours/df_{im_size}/df.shp')
data_dir = Path(data_dir)
conds = np.array(sorted([float(p.name) for p in data_dir.iterdir() if p.is_dir()]))



# actual size dist
dataset = ShapeData(path, im_size=im_size, mode='all', augment=False)

real_sizes = np.empty(len(dataset))
for i in tqdm.trange(len(dataset)):
    out = dataset[i]
    real_sizes[i] = out['class_label'].item()

moments = scipy.stats.gamma.fit(real_sizes)
xs = np.linspace(real_sizes.min(), real_sizes.max(), 1000)
ys = scipy.stats.gamma.pdf(xs, *moments)

plt.hist(real_sizes, bins=50, density=True, label='Actual distribution')
plt.plot(xs, ys, label='Fitted gamma distribution')

plt.legend()
plt.title('Size distribution')
plt.xlabel('Size')
plt.ylabel('Density')

plt.savefig(out_dir / f'size_dist.{out_ext}', **fig_kwargs)
if show_plots:
    plt.show()



ims = list()
for cond in conds:
    ims.append(load_data(data_dir / f'{cond:.2f}', dtype=bool))
ims = np.stack(ims)

sample_sizes = ims.mean(axis=(2, 3))

plt.vlines(
    (
        np.quantile(real_sizes, (1 - data_interval) / 2),
        np.quantile(real_sizes, (1 + data_interval) / 2)),
    conds.min(),
    conds.max(),
    colors=['C1', 'C1'],
    linestyles='dashed',
    label=' data interval' if data_interval == 1. else f'{data_interval * 100:.0f}% data interval'
)
plt.fill_between(
    conds,
    np.quantile(sample_sizes, (1 - sample_interval) / 2, axis=1),
    np.quantile(sample_sizes, (1 + sample_interval) / 2, axis=1),
    alpha=0.3,
    color='C0',
    label=f'{sample_interval * 100:.0f}% sample interval')
plt.plot((conds.min(), conds.max()), (conds.min(), conds.max()), 'k--', label='ideal')
plt.plot(conds, sample_sizes.mean(axis=1), c='C0', label='mean')

plt.legend()
plt.title('Size conditioning')
plt.xlabel('Condition size')
plt.ylabel('Actual size')

plt.savefig(out_dir / f'size_cond.{out_ext}', **fig_kwargs)
if show_plots:
    plt.show()
