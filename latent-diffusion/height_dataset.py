import os
from pathlib import Path
from typing import Optional, Literal

import cv2
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

import torch
import torch.utils.data
from torch.utils.data import Dataset
from torchvision import transforms


class HeightData(Dataset):

    def __init__(
        self,
        path: os.PathLike,
        im_size: int=256,
        data_im_size: int=724,
        *,
        mode: Literal['train', 'val', 'test']='train',
        val_split: float=0.1,
        test_split: float=0.1,
        dtype: Optional[torch.dtype]=None,
        seed: int=42069
    ) -> None:
        super().__init__()

        # load data
        self.path = Path(path)
        df = self._read_file()

        # remove edge data
        if data_im_size == 724:
            df = df[df['center_x'].between(64, 660) & df['center_y'].between(64, 660)]
        else:
            raise NotImplementedError(data_im_size)

        # split data
        assert mode in ['train', 'val', 'test'], 'mode must be one of "train", "val", "test"'
        assert val_split + test_split < 1, 'val_split + test_split must be less than 1'

        generator = np.random.default_rng(seed)
        datapoint_ids = df.index
        num_datapoints = len(datapoint_ids)
        evaluation_datapoints = generator.choice(datapoint_ids, int(np.ceil(num_datapoints * (val_split + test_split))), replace=False)
        train_datapoints = np.setdiff1d(datapoint_ids, evaluation_datapoints)
        val_datapoints = generator.choice(evaluation_datapoints, int(np.floor(num_datapoints * val_split)), replace=False)
        test_datapoints = np.setdiff1d(evaluation_datapoints, val_datapoints)

        if mode == 'train':
            df = df.loc[train_datapoints]
        elif mode == 'val':
            df = df.loc[val_datapoints]
        elif mode == 'test':
            df = df.loc[test_datapoints]

        # get contours
        self.contours = [np.array(list(geom.exterior.coords)) for geom in df.geometry]
        self.paths = [Path(p) for p in df['path']]
        self.offsets = df['offset'].to_numpy()

        # set attributes
        self.im_size = im_size
        self.data_im_size = data_im_size

        self.im_center = (im_size // 2, im_size // 2)
        self.data_im_center = (data_im_size // 2, data_im_size // 2)
        
        self.dtype = dtype if dtype is not None else torch.float32

        self.shape_transform = transforms.Compose([
            transforms.Normalize(mean=127.5, std=127.5),
        ])
        self.height_transform = transforms.Compose([
            # transforms.Normalize(mean=0, std=1),
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

    def get_mask(self, points: np.ndarray) -> torch.Tensor:
        im = np.zeros((self.data_im_size, self.data_im_size), dtype=np.uint8)
        im = cv2.drawContours(im, points[:, None].astype(int), -1, 255, thickness=cv2.FILLED)
        cv2.fillPoly(im, pts=[(points)[:, None].astype(int)], color=255)
        return im

    def __getitem__(self, idx):

        # load
        contour = self.contours[idx]
        offset = self.offsets[idx]
        height_map = np.load(self.paths[idx])[..., None].astype(np.float32)

        # mirror
        if np.random.rand() > 0.5:
            contour[:, 0] = self.data_im_size - contour[:, 0]
            height_map = np.flip(height_map, axis=1)

        # rotate
        angle = np.random.uniform(0, 2 * np.pi)
        rot_mat = cv2.getRotationMatrix2D(self.data_im_center, angle, 1.0)

        contour = np.concatenate((contour, np.ones((contour.shape[0], 1))), axis=1) @ rot_mat.T

        shape = self.get_mask(contour)

        # mask
        height_map = cv2.warpAffine(height_map, rot_mat, height_map.shape[1::-1], flags=cv2.INTER_LINEAR)
        height_map -= offset
        height_map[~shape.astype(bool)] = 0.
        height_map = np.clip(height_map, 0, np.inf)

        # crop
        mask_idxs = np.argwhere(shape)
        center = (mask_idxs.min(axis=0) + mask_idxs.max(axis=0)) / 2
        top_left = np.round(center - self.im_center).astype(int)
        bottom_right = top_left + self.im_size
        # assert (mask_idxs.max(axis=0) - mask_idxs.min(axis=0)).max() < self.im_size, 'contour too large'

        shape = shape[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
        height_map = height_map[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
        
        return {
            'image': self.height_transform(torch.tensor(height_map[None], dtype=self.dtype)).squeeze(0).unsqueeze(-1),
            'shape': self.shape_transform(torch.tensor(shape[None], dtype=self.dtype)).squeeze(0).unsqueeze(-1),
            'human_label': f'offset={offset:.2f}',
        }



if __name__ == '__main__':
    import matplotlib.pyplot as plt

    im_size = 128
    key = 'image'

    path = f'./data/height_contours/df_{im_size}/df.shp'

    for mode in ['test', 'val', 'train']:
        dataset = HeightData(path, im_size=im_size, mode=mode)
        print(f'{mode}: {len(dataset)}')
        break

    while True:
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        idxs = np.random.choice(len(dataset), replace=False, size=4)
        for idx, ax in zip(idxs, axes.flatten()):
            im = dataset[idx][key].cpu().numpy().squeeze(-1)
            ax.set_title(f'idx: {idx}, min: {im.min():.2f}, max: {im.max():.2f}')
            ax.matshow(im, cmap='gray')
            ax.axis('off')
            # ax.colorbar()
        fig.suptitle('Random contours')
        fig.tight_layout()
        plt.show()

    fig, axes = plt.subplots(8, 8, figsize=(40, 40))
    idx = np.random.choice(len(dataset))
    for ax in axes.flatten():
        ax.imshow(dataset[idx][key].cpu().numpy().squeeze(-1), cmap='gray')
        ax.axis('off')
    fig.suptitle('Random contour transforms')
    plt.show()





"""
Christian
Been at trifork for little over a year
Student at DTU where I'll start master thesis december
In my sparetime I like to run and cook
"""
