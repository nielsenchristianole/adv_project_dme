import os
from pathlib import Path
from typing import Optional, Literal, Union, Tuple

import cv2
import tqdm
import scipy.stats
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

import torch
import torch.utils.data
from torch.utils.data import Dataset
from torchvision import transforms

from ldm.modules.diffusionmodules.util import timestep_embedding


class ShapeData(Dataset):

    def __init__(
        self,
        path: os.PathLike,
        im_size: int=256,
        *,
        mode: Literal['train', 'val', 'test', 'all']='train',
        val_split: float=0.1,
        test_split: float=0.1,
        dtype: Optional[torch.dtype]=None,
        pix2m=2000,
        seed: int=42069,
        augment: Optional[bool]=True
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

        # get contours
        centers = np.array([list(c.coords) for c in df.minimum_bounding_circle().centroid]).squeeze(1)
        contours = [np.array(list(geom.exterior.coords)) for geom in df.geometry]
        self.contours = [(contour - center) / pix2m for contour, center in zip(contours, centers)]

        # set attributes
        self.augment = augment
        self.im_size = im_size
        self.dtype = dtype if dtype is not None else torch.float32

        self.transform = transforms.Compose([
            transforms.Lambda(lambda t: t / 255),
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
        points[:,0] = -points[:,0]
        return points

    @staticmethod
    def rotate(points: np.ndarray, angle: float) -> np.ndarray:
        c, s = np.cos(angle), np.sin(angle)
        rot = np.array([[c, -s], [s, c]])
        return points @ rot

    def get_im(self, points: np.ndarray) -> torch.Tensor:
        im = np.zeros((self.im_size, self.im_size), dtype=np.uint8)
        im = cv2.drawContours(im, (points + self.im_size / 2)[:, None].astype(int), -1, 255, thickness=cv2.FILLED)
        cv2.fillPoly(im, pts=[(points + self.im_size / 2)[:, None].astype(int)], color=255)
        return torch.tensor(im[None], dtype=self.dtype)

    def __getitem__(self, idx: int, *, mirror: Optional[bool]=None, angle: Optional[bool]=None) -> dict:
        contour = self.contours[idx]

        if self.augment:
            if mirror or ((mirror is None) and (np.random.rand() >= 0.5)):
                contour = self.mirror(contour)
            angle = angle if angle is not None else np.random.uniform(0, 2 * np.pi)
        else:
            angle = 0.
        contour = self.rotate(contour, angle)
        image = self.transform(self.get_im(contour)).squeeze(0).unsqueeze(-1)
        size = (image > 0).float().mean()[None]

        return {
            'image': image,
            'class_label': size,
            'human_label': f'size={size.item():.2f}',
        }


class SinusoidalEmbedder(torch.nn.Module):

    def __init__(
        self,
        key: str,
        emb_dim: int=256,
        max_period: Union[float,int]=1,
        integer_to_continuous: bool=False,
        use_deterministic: Optional[bool]=None,
        scale: float=1.0
    ):
        self.key = key
        self.scale = scale
        self.emb_dim = emb_dim
        self.max_period = max_period
        self.integer_to_continuous = integer_to_continuous
        self.use_deterministic = use_deterministic
        super().__init__()

    def forward(self, xc: dict) -> torch.Tensor:
        time = xc[self.key]
        if self.integer_to_continuous:
            time += torch.tensor((0.5,)) if self.use_deterministic else torch.rand((1,))

        return timestep_embedding(self.scale * time, self.emb_dim, max_period=self.max_period)

    @staticmethod
    def sample_sizes(num_samples: int) -> np.ndarray:
        """
        This dist has been fittet to the shape data
        """
        return scipy.stats.gamma.rvs(*(3.556653401460509, 0.009988185776161805, 0.03552099331432453), size=num_samples)
    
    def sample(self, num_samples: int, *, return_scalers: bool=False, conds: Optional[float]=None) -> Union[torch.Tensor, Tuple[torch.Tensor, np.ndarray]]:
        sizes = self.sample_sizes(num_samples) if conds is None else np.full(num_samples, conds).astype(float)
        time = torch.tensor(sizes).float()
        conds = self.forward({self.key: time}).unsqueeze(1)
        if return_scalers:
            return conds, sizes
        return conds

    def sanity_check(self, num_samples: Optional[int]=None) -> None:
        num_samples = num_samples if num_samples is not None else 2 * self.emb_dim
        time = torch.linspace(0, self.max_period, num_samples)[:, None]
        emb = self.forward({self.key: time})
        plt.imshow(emb.squeeze(1).numpy())
        plt.show()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # SinusoidalEmbedder('time', max_period=80.0).sanity_check()
    # quit()

    im_size = 128
    path = Path(f'./data/contours/df_{im_size}/df.shp')

    # dataset = ShapeData(path, im_size=im_size, mode='all', augment=False)

    # sizes = torch.empty(len(dataset))
    # for i in tqdm.trange(len(dataset)):
    #     out = dataset[i]
    #     sizes[i] = out['class_label'].item()
    # sizes = sizes.numpy()

    # import scipy.stats
    # # moments = scipy.stats.gamma.fit(sizes)

    # moments = (3.556653401460509, 0.009988185776161805, 0.03552099331432453)

    # # xs = np.linspace(sizes.min(), sizes.max(), 1000)
    # xs = np.linspace(0, 1, 1000)
    # ys = scipy.stats.gamma.pdf(xs, *moments)

    # plt.hist(sizes, bins=50, density=True)
    # plt.plot(xs, ys)
    # plt.show()


    # quit()


    for mode in ['test', 'val', 'train']:
        dataset = ShapeData(path, im_size=im_size, mode=mode)
        print(f'{mode}: {len(dataset)}')
        # break

    # while True:
    #     fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    #     idxs = np.random.choice(len(dataset), replace=False, size=4)
    #     for idx, ax in zip(idxs, axes.flatten()):
    #         ax.imshow(dataset[idx]['image'].cpu().numpy().squeeze(-1), cmap='gray')
    #         ax.axis('off')
    #     fig.suptitle('Random contours')
    #     fig.tight_layout()
    #     plt.show()

    # fig, axes = plt.subplots(8, 8, figsize=(40, 40))
    # idx = np.random.choice(len(dataset))
    # for ax in axes.flatten():
    #     ax.imshow(dataset[idx]['image'].cpu().numpy().squeeze(-1), cmap='gray')
    #     ax.axis('off')
    # fig.suptitle('Random contour transforms')
    # plt.show()
