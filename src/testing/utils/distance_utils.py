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
    """
    x and y are np.ndarrays
    """

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

    mmd = sample_target_dist.min(axis=1)[:num_target].mean()
    coverage = len(np.unique(sample_target_dist.argmin(axis=1)[:num_target])) / num_target
    one_nnA = (sample_target_dist.min(axis=1)[:num_target + 1] < remove_diag(sample_sample_dist[:num_target + 1, :num_target + 1]).min(axis=1)).mean()

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
