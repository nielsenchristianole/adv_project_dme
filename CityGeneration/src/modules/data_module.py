
import os
from types import SimpleNamespace as NameSpace

from torch.utils.data import DataLoader
from lightning import LightningDataModule

class DataModule(LightningDataModule):
    
    def __init__(self, 
                 dataset : any,
                 dataset_params : NameSpace,
                 dataloader_params : NameSpace):
        super().__init__()
        """
        LightningDataModule for the ear dataset. Allows for easy loading of the dataset into a Lightning model.
        Trainer.fit() will automatically call the train_dataloader() and val_dataloader() methods.
        """
        
        self.dataloader_params = dataloader_params

        self.train_dataset = dataset(**vars(dataset_params), is_train = True)
        self.val_dataset = dataset(**vars(dataset_params), is_train = False)

        if dataloader_params.num_workers == -1:
            dataloader_params.num_workers = os.cpu_count()
        assert dataloader_params.num_workers > 1, f"Number of workers should be -1 (auto) or > 1 (least 1 for validation and 1 for training). \
                                                   Currently; {dataloader_params.num_workers}"

        self.val_workers = max(1,int(dataloader_params.num_workers*dataloader_params.val_worker_ratio))
        self.train_workers = dataloader_params.num_workers - self.val_workers
        
        if self.train_workers == 0:
            raise ValueError("Use a higher number of workers. Currently;[training:", self.train_workers, "validation:", self.val_workers,"]")
       
        print(f"[DATALOADER] Workers using: train={self.train_workers}, val={self.val_workers}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset,
                          batch_size=self.dataloader_params.batch_size,
                          shuffle=True,
                          num_workers = self.train_workers,
                          persistent_workers=self.dataloader_params.persistent_workers,
                          pin_memory=self.dataloader_params.pin_memory)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset,
                          batch_size=self.dataloader_params.batch_size,
                          shuffle=False,
                          num_workers = self.val_workers,
                          persistent_workers=self.dataloader_params.persistent_workers,
                          pin_memory=self.dataloader_params.pin_memory)