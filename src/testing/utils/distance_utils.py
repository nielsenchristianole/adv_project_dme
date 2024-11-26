from typing import Callable, Tuple
from pyemd import emd
import tqdm
import numpy as np
from pathlib import Path
from functools import lru_cache
import cv2


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


@lru_cache(maxsize=32)
def get_emd_dist_matrix(
    shape: Tuple[int, int]
) -> np.ndarray:
    """
    Returns a distance matrix for EMD
    """
    xs = np.linspace(0, 1, shape[0])
    ys = np.linspace(0, 1, shape[1])
    x_idx, y_idx = map(np.ravel, np.meshgrid(xs, ys))
    vecs = np.stack((x_idx, y_idx), axis=1)
    return dist_matrix(
        vecs,
        vecs,
        L2)


def emd_dist(
    x_in: np.ndarray,
    y_in: np.ndarray,
    *,
    reduced_shape: Tuple[int, int]=(16, 16)
) -> float:
    """
    x and y are floats
    """
    x = cv2.resize(x_in.astype(np.float64), reduced_shape, interpolation=cv2.INTER_LINEAR)
    y = cv2.resize(y_in.astype(np.float64), reduced_shape, interpolation=cv2.INTER_LINEAR)
    dist_matrix = get_emd_dist_matrix(x.shape)
    x, y = map(np.ravel, (x, y))
    x, y = x / x.sum(), y / y.sum()
    return emd(
        x,
        y,
        dist_matrix)


def dist_matrix(
    x: np.ndarray,
    y: np.ndarray,
    dist_fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    verbose: bool=False,
    symmetric: bool=False
) -> np.ndarray:

    dist = np.empty((len(x), len(y)))
    if symmetric:
        np.fill_diagonal(dist, 0.)

    if verbose:
        if symmetric:
            pbar = tqdm.tqdm(total=len(x) * (len(y) - 1) // 2, leave=False)
        else:
            pbar = tqdm.tqdm(total=len(x) * len(y), leave=False)

    for i, x_i in enumerate(x):
        y_enum = y[:i] if symmetric else y
        for j, y_j in enumerate(y_enum):
            dist[i, j] = dist_fn(x_i, y_j)
            if symmetric:
                dist[j, i] = dist[i, j]
            if verbose:
                pbar.update(1)
    return dist


def remove_diag(x: np.ndarray) -> np.ndarray:
    return x[~np.eye(x.shape[0], dtype=bool)].reshape(x.shape[0], -1)


def calculate_stats(
    samples: np.ndarray,
    targets: np.ndarray,
    dist_func: Callable[[np.ndarray, np.ndarray], float],
    verbose: bool=True
) -> Tuple[float, float, float]:
    """
    returns mmd, coverage, one_nnA
    """

    num_target = len(targets)
    assert len(samples) >= 1 + num_target
    samples = samples[:1 + num_target]
    targets = targets[:1 + num_target]

    if verbose: pbar = tqdm.tqdm(total=3, leave=False)
    sample_target_dist = dist_matrix(samples, targets, dist_func, verbose=verbose)
    if verbose: pbar.update(1)
    sample_sample_dist = dist_matrix(samples, samples, dist_func, verbose=verbose, symmetric=True)
    if verbose: pbar.update(1)
    target_target_dist = dist_matrix(targets, targets, dist_func, verbose=verbose, symmetric=True)
    if verbose: pbar.update(1)
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


if __name__ == "__main__":

    im1 = np.random.exponential(size=(64, 64))
    im2 = np.random.exponential(size=(64, 64))

    d = emd_dist(im1, im2, reduced_shape=(32, 32))
    print(d)

    quit()
