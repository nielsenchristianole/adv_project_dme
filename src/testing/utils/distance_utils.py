from typing import Callable, Tuple

import numpy as np
from pathlib import Path


IM_SIZE = 128


def L2(
    x: np.ndarray,
    y: np.ndarray
) -> float:
    """
    x and y are floats
    """
    return np.linalg.norm(x - y)


def iou_dist(
    x: np.ndarray,
    y: np.ndarray
) -> float:
    """
    x and y are bools
    """
    intersection = x & y
    union = x | y
    return 1 - intersection.sum() / union.sum()


def dice_dist(
    x: np.ndarray,
    y: np.ndarray
) -> float:
    """
    x and y are bools
    """
    intersection = x & y
    return 1 - 2 * intersection.sum() / (x.sum() + y.sum())


def dist_matrix(
    x: np.ndarray,
    y: np.ndarray,
    dist_fn: Callable[[np.ndarray, np.ndarray], float]
) -> np.ndarray:

    dist = np.empty((len(x), len(y)))
    for i, x_i in enumerate(x):
        for j, y_j in enumerate(y):
            dist[i, j] = dist_fn(x_i, y_j)
    return dist


def remove_diag(x: np.ndarray) -> np.ndarray:
    return x[~np.eye(x.shape[0], dtype=bool)].reshape(x.shape[0], -1)


def calculate_stats(samples: np.ndarray, targets: np.ndarray, dist_func: Callable[[np.ndarray, np.ndarray], float]) -> Tuple[float, float, float]:
    """
    returns mmd, coverage, one_nnA
    """

    num_target = len(targets)
    assert len(samples) >= 1 + num_target

    sample_target_dist = dist_matrix(samples, targets, dist_func)
    sample_sample_dist = dist_matrix(samples, samples, dist_func)
    target_target_dist = dist_matrix(targets, targets, dist_func)
    target_sample_dist = sample_target_dist[:num_target - 1].T

    mmd = sample_target_dist.min(axis=1)[:num_target].mean()
    coverage = len(np.unique(sample_target_dist.argmin(axis=1)[:num_target])) / num_target

    # for sample classification
    extra_nn_dist = sample_target_dist.min(axis=1)[:num_target + 1]
    intra_nn_dist = remove_diag(sample_sample_dist[:num_target + 1, :num_target + 1]).min(axis=1)
    one_nnA_sample = (extra_nn_dist < intra_nn_dist).sum() + 0.5 * (extra_nn_dist == intra_nn_dist).sum()

    # for target classification
    extra_nn_dist = target_sample_dist.min(axis=1)[:num_target]
    intra_nn_dist = remove_diag(target_target_dist).min(axis=1)
    one_nnA_target = (extra_nn_dist < intra_nn_dist).sum() + 0.5 * (extra_nn_dist == intra_nn_dist).sum()

    one_nnA = (one_nnA_sample + one_nnA_target) / (2 * num_target + 1)

    if False:
        extra_cov = sample_target_dist[:num_target + 1].argmin(axis=1)
        intra_cov = remove_diag(sample_sample_dist[:num_target + 1, :num_target + 1]).argmin(axis=1)
        intra_cov += (intra_cov >= np.arange(num_target + 1))
        closest_match = np.where((extra_nn_dist < intra_nn_dist)[:, None, None], targets[extra_cov], samples[intra_cov])

        import matplotlib.pyplot as plt
        idx = 0
        fig, axs = plt.subplots(2, 1, figsize=(4, 8))
        axs[0].imshow(samples[idx])
        axs[1].imshow(closest_match[idx])


    return mmd, coverage, one_nnA


def uniform_to_height(u: np.ndarray, *, mean: float=300.0) -> np.ndarray:
    """
    Transform uniform samples to height samples
    u ~ U(0, 1)
    height ~ Exp(mean)
    """
    return - mean * np.log(1 - u)


def load_data(_dir: Path, dtype=float, im_size: int=IM_SIZE) -> np.ndarray:
    paths = sorted(list(_dir.iterdir()))
    arr = np.empty((len(paths), im_size, im_size), dtype=dtype)
    for i, sample_path in enumerate(paths):
        arr[i] = np.load(sample_path).astype(dtype)
    return arr
