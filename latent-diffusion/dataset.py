from os import PathLike
from typing import Literal, Optional
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from ldm.modules.diffusionmodules.util import timestep_embedding


class AutoEncoderDataset(Dataset):
    current_data: pd.DataFrame

    def __init__(
        self,
        csv_file: PathLike,
        data_root_dir: PathLike,
        val: float = 0.1,
        test: float = 0.2,
        seed: int = 69420,
        vertebrae_emb_dim: int = 256,
        mode: Literal['train', 'val', 'test', 'trainandval', 'all'] = 'train',
        return_everything: bool = False,
        im_resize: Optional[int] = None,
        normalize: bool = False,
        use_deterministic_vertebrae_emb: bool = False,
        mask_resize: Optional[int] = None
    ):
        
        self.return_everything = return_everything
        self.use_deterministic_vertebrae_emb = use_deterministic_vertebrae_emb

        self.data_root_dir = Path(data_root_dir)
        self.columns = (
            'patient_id', 'im_id', 'patient_ix', 'corrupted_path',
            'tissue_path', 'mask_path', 'ct_path', 'vertebrae',
            'num_bg', 'num_inside', 'num_tissue', 'num_mask')

        self.vertebrae_emb_dim = vertebrae_emb_dim
        self.data = pd.read_csv(csv_file)

        generator = np.random.default_rng(seed)
        patient_ids = self.data['patient_id'].unique()
        num_patients = len(patient_ids)
        evaluation_patients = generator.choice(patient_ids, int(np.ceil(num_patients * (val + test))), replace=False)
        train_patients = np.setdiff1d(patient_ids, evaluation_patients)
        val_patient = generator.choice(evaluation_patients, int(np.floor(num_patients * val)), replace=False)
        test_patients = np.setdiff1d(evaluation_patients, val_patient)

        self.train_data = self.data[self.data['patient_id'].apply(lambda x: x in train_patients)].reset_index(drop=True)
        self.val_data = self.data[self.data['patient_id'].apply(lambda x: x in val_patient)].reset_index(drop=True)
        self.test_data = self.data[self.data['patient_id'].apply(lambda x: x in test_patients)].reset_index(drop=True)

        self.change_mode(mode)

        _transforms = [transforms.ToTensor()]
        _transforms_without_norm = [transforms.ToTensor()]
        if (im_resize and im_resize != 256):
            _transforms.append(transforms.Resize((im_resize, im_resize)))
        if (mask_resize and mask_resize != 64):
            _transforms_without_norm.append(transforms.Resize((mask_resize, mask_resize)))
        if normalize:
            _transforms.append(transforms.Normalize(mean=[0.5], std=[0.5]))
        self.transform = transforms.Compose(_transforms)
        self.transform_without_norm = transforms.Compose(_transforms_without_norm)

    def __len__(self):
        return len(self.current_data)

    def change_mode(self, mode: Literal['train', 'val', 'test']):
        if mode == 'train':
            self.current_data = self.train_data
        elif mode == 'val':
            self.current_data = self.val_data
        elif mode == 'test':
            self.current_data = self.test_data
        elif mode == 'trainandval':
            self.current_data = pd.concat([self.train_data, self.val_data], ignore_index=True)
        elif mode == 'all':
            self.current_data = self.data
        else:
            raise ValueError('mode must be one of "train", "val", or "test"')
        return self
    
    def emb_vertebrae(self, vertebrae: int) -> torch.Tensor:
        rand = torch.tensor((0.5,)) if self.use_deterministic_vertebrae_emb else torch.rand((1,))
        time = (rand + vertebrae) * 2 + 1
        return timestep_embedding(time, self.vertebrae_emb_dim, max_period=100)
    
    def read_image(self, path) -> torch.Tensor:
        im = cv2.imread(str(self.data_root_dir / path), cv2.IMREAD_ANYCOLOR)
        return self.transform(im).squeeze(0).unsqueeze(-1) # x.permute(0, 3, 1, 2)

    def __getitem__(self, idx):
        item = self.current_data.iloc[idx]

        vert = item['vertebrae']
        batch = dict(human_label=str(vert), class_label=self.emb_vertebrae(vert))

        batch['image'] = self.read_image(item['ct_path'])

        if self.return_everything:
            batch['corrupted'] = self.read_image(item['corrupted_path'])
            batch['tissue'] = self.read_image(item['tissue_path'])
            batch['mask'] = self.read_image(item['mask_path'])

        return batch
    
    def from_api(
        self,
        corrupted_image: np.ndarray,
        mask_image: np.ndarray,
        vertebrae: int,
        *,
        num_monte_carlo: int = 1,
        device='cuda'
    ):
        assert self.use_deterministic_vertebrae_emb, 'should_be_deterministic'

        batch = dict(human_label=num_monte_carlo*[str(vertebrae)])

        label = self.emb_vertebrae(vertebrae)
        batch['class_label'] = label[None, ...].repeat(num_monte_carlo, 1, 1)

        im = self.transform(corrupted_image).to(device)
        batch['corrupted'] = im[..., None].repeat(num_monte_carlo, 1, 1, 1)

        mask = self.transform_without_norm(mask_image).unsqueeze(0).to(device)
        mask = 1 - (mask > 0).float() # we want to occlude, where the resized image is partially occluded
        batch['mask'] = mask.repeat(num_monte_carlo, 1, 1, 1)
        
        return batch


class ReadDictEmbedder(torch.nn.Module):

    def __init__(self, key: str):
        self.key = key
        super().__init__()

    def forward(self, xc: dict) -> torch.Tensor:
        return xc[self.key]




if __name__ == '__main__':
    import tqdm

    dataset = AutoEncoderDataset(
        csv_file='../data/dataframe.csv',
        data_root_dir='../',
        im_resize=128,
        normalize=True)
    
    print('number of train:', len(dataset))
    pbar = tqdm.trange(len(dataset), leave=False)
    for i in pbar:
        batch = dataset[i]
        pbar.set_description(f"Processed {i} samples with shape {batch['image'].shape}")

    dataset.change_mode('val')
    print('number of val:', len(dataset))
    pbar = tqdm.trange(len(dataset), leave=False)
    for i in pbar:
        batch = dataset[i]
        pbar.set_description(f"Processed {i} samples with shape {batch['image'].shape}")
    
    dataset.change_mode('test')
    print('number of test:', len(dataset))
    pbar = tqdm.trange(len(dataset), leave=False)
    for i in pbar:
        batch = dataset[i]
        pbar.set_description(f"Processed {i} samples with shape {batch['image'].shape}")
    
    dataset.change_mode('trainandval')
    print('number of train and val:', len(dataset))
    pbar = tqdm.trange(len(dataset), leave=False)
    for i in pbar:
        batch = dataset[i]
        pbar.set_description(f"Processed {i} samples with shape {batch['image'].shape}")
    
    dataset.change_mode('all')
    print('number of all:', len(dataset))
    pbar = tqdm.trange(len(dataset), leave=False)
    for i in pbar:
        batch = dataset[i]
        pbar.set_description(f"Processed {i} samples with shape {batch['image'].shape}")
