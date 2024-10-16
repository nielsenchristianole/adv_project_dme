import os
from pathlib import Path
from typing import List, Tuple, Union, Iterable, Optional

import cv2
import tqdm
import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import shapely
import scipy.spatial

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ShapeData(Dataset):

    def __init__(
        self,
        path: os.PathLike,
        im_size: int=256,
        *,
        dtype: Optional[torch.dtype]=None,
        device: Optional[torch.device]=None,
        pix2m=2000
    ) -> None:
        super().__init__()

        # load data
        self.path = Path(path)
        df = self._read_file()

        # sort out contours larger than the image
        radii = df.minimum_bounding_radius()
        mask = (radii <= im_size * pix2m / 2)
        if not mask.any():
            raise ValueError('No contours found within the specified range')
        df = df[mask]

        # get contours
        centers = np.array([list(c.coords) for c in df.minimum_bounding_circle().centroid]).squeeze(1)
        contours = [np.array(list(geom.exterior.coords)) for geom in df.geometry]
        self.contours = [(contour - center) / pix2m for contour, center in zip(contours, centers)]

        # set attributes
        self.im_size = im_size
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype if dtype is not None else torch.float32

        self.transform = transforms.Compose([
            transforms.Normalize(mean=127.5, std=127.5),
        ])

    def _read_file(self) -> gpd.GeoDataFrame:
        assert self.path.exists(), 'File does not exist'
        try:
            return gpd.read_file(filename=str(self.path))
        except AttributeError:
            import fiona
            print('force fiona to 1.9.6, instead of current:', fiona.__version__)
            raise ImportError('fiona 1.9.6 is required')

    def __len__(self):
        return len(self.contours)

    @staticmethod
    def mirror(points: np.ndarray) -> np.ndarray:
        return points[::-1]
    
    @staticmethod
    def rotate(points: np.ndarray, angle: float) -> np.ndarray:
        c, s = np.cos(angle), np.sin(angle)
        rot = np.array([[c, -s], [s, c]])
        return points @ rot

    def get_im(self, points: np.ndarray) -> torch.Tensor:
        im = np.zeros((self.im_size, self.im_size), dtype=np.uint8)
        im = cv2.drawContours(im, (points + self.im_size / 2)[:, None].astype(int), -1, 255, thickness=cv2.FILLED)
        cv2.fillPoly(im, pts=[(points + self.im_size / 2)[:, None].astype(int)], color=255)
        return torch.tensor(im[None], dtype=self.dtype, device=self.device)

    def __getitem__(self, idx):
        contour = self.contours[idx]

        if np.random.rand() > 0.5:
            contour = self.mirror(contour)
        angle = np.random.uniform(0, 2 * np.pi)
        contour = self.rotate(contour, angle)
        return self.transform(self.get_im(contour))



if __name__ == '__main__':
    import matplotlib.pyplot as plt

    path = Path('./data/contours/df_prefilter/df.shp')
    dataset = ShapeData(path, im_size=64)
    

    while True:
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        idxs = np.random.choice(len(dataset), replace=False, size=4)
        for idx, ax in zip(idxs, axes.flatten()):
            ax.imshow(dataset[idx].cpu().numpy().squeeze(0), cmap='gray')
            ax.axis('off')
        fig.suptitle('Random contours')
        fig.tight_layout()
        plt.show()

    fig, axes = plt.subplots(8, 8, figsize=(40, 40))
    idx = np.random.choice(len(dataset))
    for ax in axes.flatten():
        ax.imshow(dataset[idx].cpu().numpy().squeeze(0), cmap='gray')
        ax.axis('off')
    fig.suptitle('Random contour transforms')
    plt.show()
