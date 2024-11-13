import tqdm
import numpy as np
import tabulate
import matplotlib.pyplot as plt
from pathlib import Path

from src.testing.utils.distance_utils import L2, iou_dist, dice_dist, calculate_stats, uniform_to_height, load_data


sample_dir = './data/samples'
out_dir = './results/shapes_and_heights'
table_kwargs = dict(
    tablefmt='latex',
    floatfmt='.3f',
)



sample_dir = Path(sample_dir)
out_dir = Path(out_dir)
out_dir.mkdir(exist_ok=True, parents=True)
pbar = tqdm.tqdm(total=10)


# load samples
test_shapes = load_data(sample_dir / 'test/shapes', dtype=bool)
test_heights = load_data(sample_dir / 'test/heights', dtype=float)

train_shapes = load_data(sample_dir / 'train/shapes', dtype=bool)
train_heights = load_data(sample_dir / 'train/heights', dtype=float)

sample_shapes = load_data(sample_dir / 'shapes', dtype=bool)
sample_heights = load_data(sample_dir / 'heights', dtype=float)
pbar.update(1)

raw_table_str = r"""
\begin{table}[h!]
\centering

\begin{tabular}{lrrrrrr}
    \toprule
    \multirow{2}{*}{Method} & \multicolumn{2}{c}{MMD ($\downarrow$)} & \multicolumn{2}{c}{COV ($\uparrow$)} & \multicolumn{2}{c}{1nn ($\downarrow$)} \\
    \cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}
     & __dist__ & __dist__ & __dist__ \\
    \midrule
    __table__
    \bottomrule
    \end{tabular}
    \caption{__caption__}
    \label{tab:__label__}
\end{table}
"""

# v1 = calculate_stats(sample_heights, test_heights, L2)

# calculate shape stats
shape_tabular_vals = []
v1 = calculate_stats(sample_shapes, test_shapes, iou_dist)
pbar.update(1)
v2 = calculate_stats(sample_shapes, test_shapes, dice_dist)
pbar.update(1)

row = ['Samples']
for vals in zip(v1, v2):
    row.extend(vals)
shape_tabular_vals.append(row)

pbar.update(1)
v1 = calculate_stats(train_shapes, test_shapes, iou_dist)
pbar.update(1)
v2 = calculate_stats(train_shapes, test_shapes, dice_dist)

row = ['Train subsample']
for vals in zip(v1, v2):
    row.extend(vals)
shape_tabular_vals.append(row)

# calculate height stats
height_tabular_vals = []
v1 = calculate_stats(sample_heights, test_heights, L2)
pbar.update(1)
v2 = calculate_stats(uniform_to_height(sample_heights), uniform_to_height(test_heights), L2)
pbar.update(1)
row = ['Samples']
for vals in zip(v1, v2):
    row.extend(vals)
height_tabular_vals.append(row)

v1 = calculate_stats(train_heights, test_heights, L2)
pbar.update(1)
v2 = calculate_stats(uniform_to_height(train_heights), uniform_to_height(test_heights), L2)
pbar.update(1)
row = ['Train subsample']
for vals in zip(v1, v2):
    row.extend(vals)
height_tabular_vals.append(row)

pbar.close()


# generate tables
def save_data(shape_tabular_vals, height_tabular_vals, table_kvargs) -> None:

    shape_tabular_str = tabulate.tabulate(shape_tabular_vals, **table_kvargs).split(r'\hline')[1]
    height_tabular_str = tabulate.tabulate(height_tabular_vals, **table_kvargs).split(r'\hline')[1]

    # write tables
    shape_table_str = raw_table_str.replace('__dist__', 'iou & dice').replace('__table__', shape_tabular_str).replace('__caption__', 'Shape metrics').replace('__label__', 'tab:shape_results')
    with open(out_dir / 'shape_table.tex', 'w') as f:
        f.write(shape_table_str)

    height_table_str = raw_table_str.replace('__dist__', r'$L2_{\text{uni}}$ & $L2_{\text{exp}}$').replace('__table__', height_tabular_str).replace('__caption__', 'Height metrics').replace('__label__', 'tab:height_results')
    with open(out_dir / 'height_table.tex', 'w') as f:
        f.write(height_table_str)


save_data(shape_tabular_vals, height_tabular_vals, table_kwargs)
import pdb
pdb.set_trace()
# update table_kwargs and rerun save_data
