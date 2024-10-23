import os
from pathlib import Path
from typing import Optional, Literal

import cv2
import tqdm
import numpy as np
import pandas as pd
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
        im_size: int=128,
        data_im_size: int=724,
        *,
        root_dir: Optional[os.PathLike]=None,
        mode: Literal['train', 'val', 'test', 'all']='train',
        val_split: float=0.1,
        test_split: float=0.1,
        dtype: Optional[torch.dtype]=None,
        seed: int=42069,
        height_mean: float=300.0,
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
        elif mode == 'all':
            pass
        else:
            raise ValueError(f'Unknown mode: {mode}')

        if root_dir is None:
            root_dir = './'
        root_dir = Path(root_dir)
        self.paths = [root_dir / p for p in df['path']]

        # get contours
        self.contours = [np.array(list(geom.exterior.coords)) for geom in df.geometry]
        self.offsets = df['offset'].to_numpy()
        self.centers = df[['center_x', 'center_y']].to_numpy()

        # set attributes
        self.im_size = im_size
        self.data_im_size = data_im_size

        self.im_center = (im_size // 2, im_size // 2)
        self.data_im_center = (data_im_size // 2, data_im_size // 2)

        self.dtype = dtype if dtype is not None else torch.float32

        self.shape_transform = transforms.Compose([
            transforms.Lambda(lambda t: t / 255),
        ])
        self.height_transform = transforms.Compose([
            transforms.Lambda(lambda t: 1 - torch.exp(-t / height_mean)),
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

    def get_mask(self, points: np.ndarray) -> np.ndarray:
        im = np.zeros((self.im_size, self.im_size), dtype=np.uint8)
        im = cv2.drawContours(im, points[:, None].astype(int), -1, 255, thickness=cv2.FILLED)
        cv2.fillPoly(im, pts=[(points)[:, None].astype(int)], color=255)
        return im

    def __getitem__(self, idx, *, mirror: Optional[bool]=None, angle: Optional[float]=None):

        # load
        contour = self.contours[idx]
        offset = self.offsets[idx]
        height_map = np.load(self.paths[idx])[..., None].astype(np.float32)
        center = self.centers[[idx]]

        # mirror
        if mirror or ((mirror is None) and (np.random.rand() > 0.5)):
            contour[:, 0] = self.data_im_size - contour[:, 0]
            center[:, 0] = self.data_im_size - center[:, 0]
            height_map = np.flip(height_map, axis=1)

        # crop
        im_center = np.array(self.im_center)[None]
        contour += im_center - center
        top_left = np.round(center - np.array(im_center)[:, None]).astype(int).ravel()
        bottom_right = top_left + self.im_size
        height_map = height_map[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        # rotate
        angle = angle if angle is not None else np.random.uniform(0, 2 * np.pi)
        rot_mat = cv2.getRotationMatrix2D(self.im_center, angle, 1.0)

        contour = np.concatenate((contour, np.ones((contour.shape[0], 1))), axis=1) @ rot_mat.T

        shape = self.get_mask(contour)

        # mask
        height_map = cv2.warpAffine(height_map[..., 0], rot_mat, (self.im_size, self.im_size), flags=cv2.INTER_LINEAR)
        height_map -= offset
        # height_map_unmasked = height_map.copy()
        height_map[~shape.astype(bool)] = 0.
        height_map = np.clip(height_map, 0, np.inf)

        return {
            'image': self.height_transform(torch.tensor(height_map[None], dtype=self.dtype)).squeeze(0).unsqueeze(-1),
            'shape': self.shape_transform(torch.tensor(shape[None], dtype=self.dtype)).squeeze(0).unsqueeze(-1),
            'human_label': f'{self.paths[idx]}\n{self.centers[idx]}',
            # 'contour': [contour],
            # 'center_coord': self.centers[idx],
            # 'height_map_unmasked': height_map_unmasked
        }



if __name__ == '__main__':
    import matplotlib.pyplot as plt

    im_size = 128
    key = 'image'

    path = f'./data/height_contours/df_{im_size}/df.shp'
    dataset = HeightData(path, im_size=im_size, mode='all')

    # data = list()
    # data_raw = list()
    # fig, ax = plt.subplots(2)
    # for i in tqdm.trange(len(dataset), desc='getting height hist'):
    #     out = dataset.__getitem__(i, mirror=False, angle=0.)
    #     im = out['image'].cpu().numpy().squeeze(-1).ravel()
    #     mask = (out['shape'].cpu().numpy().squeeze(-1).ravel() > 0)
    #     data.append(
    #         im[mask])
    #     data_raw.append(im)
    # data = np.concatenate(data)
    # data_raw = np.concatenate(data_raw)
    # ax[0].hist(data)
    # ax[0].hist(np.exp(-data))
    # plt.show()

    for i in [487, 3968, 3969, 7524]:
        fig, ax = plt.subplots(2)
        ax = ax.flatten()
        out = dataset.__getitem__(i)
        im = out[key].cpu().numpy().squeeze(-1)
        ax[0].set_title(f'idx: {i}, min: {im.min():.2f}, max: {im.max():.2f}\nlabel: {out["human_label"]}')
        ax[0].matshow(im, cmap='gray')
        ax[0].axis('off')
        ax[1].matshow(out['height_map_unmasked'], cmap='gray')
        plt.show()
    # errors = list()
    # for angle in (pbar := tqdm.tqdm(np.linspace(0, 2 * np.pi, 10, endpoint=False), 'angle', leave=True)):
    #     for mirror in tqdm.tqdm([True, False], 'mirror', leave=False):
    #         for idx in tqdm.trange(len(dataset), desc='idx', leave=False):
    #             try:
    #                 data = dataset.__getitem__(idx, mirror=mirror, angle=angle)
    #                 assert data['image'].shape == (im_size, im_size, 1)
    #                 assert data['shape'].shape == (im_size, im_size, 1)
    #             except Exception as e:
    #                 errors.append((angle, mirror, idx, e))
    #                 pbar.set_postfix_str(f'Errors: {len(errors)}')
    # if errors:
    #     df = pd.DataFrame(errors, columns=['angle', 'mirror', 'idx', 'error'])
    #     series = (df.groupby('idx').count()['error'] > 0)
    #     print(sorted(series[series].index.to_list()))
    #     print()
    #     import pdb; pdb.set_trace()

    for mode in ['test', 'val', 'train']:
        dataset = HeightData(path, im_size=im_size, mode=mode)
        print(f'{mode}: {len(dataset)}')
        break

    while True:
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        idxs = np.random.choice(len(dataset), replace=False, size=4)
        for idx, ax in zip(idxs, axes.flatten()):
            out = dataset[idx]
            im = out[key].cpu().numpy().squeeze(-1)
            ax.set_title(f'idx: {idx}, min: {im.min():.2f}, max: {im.max():.2f}\nlabel: {out["human_label"]}')
            ax.matshow(im, cmap='gray')
            ax.axis('off')
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
