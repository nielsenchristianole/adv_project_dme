import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json

import pandas as pd
import numpy as np
import tqdm

from torchvision import transforms as T
from PIL import Image

class CityDataset(Dataset):
    
    NORMALIZE_VARIANCE = 500
    
    def __init__(self,
                 unprocessed_data_path,# : str|Path,
                 processed_data_path,#: str|Path,
                 min_cities : int = 2,
                 max_cities : int = 100,
                 img_size : int = 256,
                 cutout_overlap : int = 128,
                 smoothing : int = 20,
                 verbose : bool = True,
                 data_split_seed : int = 42,
                 max_epoch_length : int = 10000,
                 val_split : float = 0.1,
                 is_train : bool = True):
        
        self.unprocessed_data_path = Path(unprocessed_data_path)
        self.processed_data_path = Path(processed_data_path)
        
        self.min_cities = min_cities
        self.max_cities = max_cities
        self.img_size = img_size
        self.cutout_overlap = cutout_overlap
        
        self.smoothing = smoothing
        
        self.verbose = verbose
        
        self.is_train = is_train
        self.val_split = val_split
        self.data_split_seed = data_split_seed

        self.max_epoch_length = max_epoch_length
        
        suffix = "train" if is_train else "val"    
        self.split_file = self.processed_data_path / f"citydata_split_info.json"

        use_existing = True
        if self.split_file.exists():
            with self.split_file.open("r") as f:
                split_info = json.load(f)
                
                min_cities = split_info["min_cities"]
                max_cities = split_info["max_cities"]
                cutout_overlap = split_info["cutout_overlap"]
                img_size = split_info["img_size"]
                smoothing = split_info["smoothing"]
                
                if min_cities != self.min_cities or \
                   max_cities != self.max_cities or \
                   cutout_overlap != self.cutout_overlap or \
                   img_size != self.img_size or \
                   smoothing != self.smoothing:
                    
                    print("[STATUS] Cache data does not match current data settings. Recomputing data extraction")
                    use_existing = False
                else:
                    use_existing = True
        else:
            use_existing = False
            
        if not use_existing:
            
        
            self.processed_data_path.mkdir(parents=True, exist_ok=True)
            (self.processed_data_path / "train").mkdir(parents=True, exist_ok=True)
            (self.processed_data_path / "val").mkdir(parents=True, exist_ok=True)
            
            save_path_train, save_path_val = self._preprocess_and_save_data()
            
            with self.split_file.open("w") as f:
                json.dump({"min_cities" : self.min_cities,
                           "max_cities" : self.max_cities,
                           "img_size" : self.img_size,
                           "smoothing" : self.smoothing,
                           "cutout_overlap" : self.cutout_overlap,
                           "paths_train" : str(save_path_train), 
                           "paths_val" : str(save_path_val)}, f)
                
        self._load_data_info()
        
        if verbose:
            print(f"[DATA INFO, {suffix}] Number of datapoints: ", len(self.height_paths))
        
    def __len__(self):
    
        # return len(self.height_paths)
        return min(len(self.height_paths), self.max_epoch_length)

    def __getitem__(self, idx : int):
        
        height = np.load(self.height_paths[idx])
        heatmap = np.load(self.heatmap_paths[idx])
        
        height, heatmap = self._prepare(height, heatmap)    
        
        return height, heatmap
    
    def _prepare(self, img : np.ndarray, heatmap : np.ndarray):
        
        diag_r = int(np.sqrt(2) * (self.img_size / 2))
        
        
        img = Image.fromarray(img)
        heatmap = Image.fromarray(heatmap)
        
        if self.is_train:
            rot_angle = np.random.randint(0, 360)
        
            img = img.rotate(rot_angle, resample=Image.BILINEAR)
            heatmap = heatmap.rotate(rot_angle, resample=Image.BILINEAR)
        
            if np.random.rand() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                heatmap = heatmap.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Extract the img_size x img_size image
        half_w = int(self.img_size / 2)
        img = img.crop((diag_r - half_w, diag_r - half_w, diag_r + half_w, diag_r + half_w))
        heatmap = heatmap.crop((diag_r - half_w, diag_r - half_w, diag_r + half_w, diag_r + half_w))
        
        # center the cities
        
        img = T.ToTensor()(img)/self.NORMALIZE_VARIANCE
        heatmap = T.ToTensor()(heatmap)
        
        heatmap = heatmap + 1e-10
        heatmap = heatmap / heatmap.sum()
        
        return img, heatmap
        
    def _load_data_info(self):
        """
        Loads the .txt files at 
        """
        # load the split file
        with self.split_file.open("r") as f:
            data = json.load(f)
            
            paths_train = data["paths_train"]
            paths_val = data["paths_val"]
            
            paths = paths_train if self.is_train else paths_val
            
        # Load the paths from the .txt file with the format:
        # (path_to_img, path_to_heatmap)
        # (path_to_img, path_to_heatmap)
        # ...
        with open(paths, "r") as f:
            data_paths = f.readlines()
            
        self.height_paths = []
        self.heatmap_paths = []
        
        for path in data_paths:
            path = path.split()
            self.height_paths.append(path[0])
            self.heatmap_paths.append(path[1])
            
        
    
    def _preprocess_and_save_data(self):
        """
        This function does the following.
        For each datapoint. Compute n x n image extractions from the data.
        n is calculates as following:
           - diag_r = sqrt(2) * (img_size / 2)
           - n = ( data_size - 2*diag_r ) % img_size
        
        Then extract img_size x img_size
        
        If the extracted image contains any cities. We store the following in an array
          - (path_idx, x, y) where x and y is the center in the original image
        """
        
        height_train_paths, city_train_paths, height_val_paths, city_val_paths = self._extract_city_and_height_paths()
        
        data_paths = {"train" : [], "val" : []}
        
        for suffix, height_paths, city_paths in tqdm.tqdm([("train", height_train_paths, city_train_paths),
                                                           ("val", height_val_paths, city_val_paths)],
                                                          desc="Extracting data",
                                                          total=2,
                                                          leave=False):
            for i, (height_path, city_path) in tqdm.tqdm(enumerate(zip(height_paths, city_paths)),
                                                        total=len(height_paths),
                                                        desc="Extracting data",
                                                        leave=False):
                    
                img_raw = np.load(height_path)
                cities = pd.read_csv(city_path)
                
                diag_r = int(np.sqrt(2) * (self.img_size / 2))
                half_w = int(self.img_size / 2)
                
                # Extract the cities from the image
                for x in range(diag_r + half_w, img_raw.shape[0] - diag_r, self.img_size - self.cutout_overlap):
                    for y in range(diag_r + half_w, img_raw.shape[1] - diag_r, self.img_size - self.cutout_overlap):
                        
                        city_extract = cities[(cities["T_x"] > x - half_w) & (cities["T_x"] < x + half_w) &
                                            (cities["T_y"] > y - half_w) & (cities["T_y"] < y + half_w) &
                                            (cities["type"] != "administrative")]
                        
                            
                        if len(city_extract) < self.min_cities or \
                        len(city_extract) > self.max_cities:
                            continue
                        
                        img, heatmap = self._preprocess(img_raw, cities, x, y)
                        
                        img_path = self.processed_data_path / suffix / f"{i}_{x}_{y}_img.npy"
                        heatmap_path = self.processed_data_path / suffix / f"{i}_{x}_{y}_heatmap.npy"
                        
                        np.save(img_path, img)
                        np.save(heatmap_path, heatmap)
                        
                        data_paths[suffix].append((img_path, heatmap_path))
                        
        # Save the data to a .txt file
        save_path_train = self.processed_data_path / "data_paths_train.txt"
        save_path_val = self.processed_data_path / "data_paths_val.txt"

        with open(save_path_train, "w") as f:
            for path in data_paths["train"]:
                f.write(f"{path[0]} {path[1]}\n")
                
        with open(save_path_val, "w") as f:
            for path in data_paths["val"]:
                f.write(f"{path[0]} {path[1]}\n")
                  
        return save_path_train, save_path_val
    
    def _preprocess(self, 
                    img : np.ndarray,
                    cities : pd.DataFrame,
                    x : int,
                    y : int):
        
        diag_r = int(np.sqrt(2) * (self.img_size / 2))
    
        img = Image.fromarray(img)
    
        # Extract 2x diag_r
        img = img.crop((x - diag_r, y - diag_r, x + diag_r, y + diag_r))
        
        cities = cities[(cities["T_x"] > x - diag_r) & (cities["T_x"] < x + diag_r) &
                        (cities["T_y"] > y - diag_r) & (cities["T_y"] < y + diag_r) &
                        (cities["type"] != "administrative")]
        cities = np.array([cities["T_x"] - (x - diag_r), cities["T_y"] - (y - diag_r)]).T
        
        img = np.array(img)
        
        # Apply water mask
        water_mask = img < 0
        img[water_mask] = 0
        
        # Convert to heatmap
        heatmap = np.zeros((diag_r*2, diag_r*2)) + 1e-10
        
        for city in cities:
            heat = np.exp(-((np.arange(diag_r*2) - city[0]) ** 2 + 
                            (np.arange(diag_r*2)[:,None] - city[1]) ** 2) /
                            (2 * self.smoothing ** 2))
            
            heat[water_mask] = 0
            heat += 1e-10
            heatmap += heat / heat.sum() 
    
        heatmap /= heatmap.sum()
            
        heatmap = torch.tensor(heatmap).float()
        return img, heatmap
        
        
    def _extract_city_and_height_paths(self):# -> tuple[list[Path], list[Path]]:
        """
        Extract the paths to the city and height data
        """
        
        # Find all paths that end with ".npy"
        all_height_paths = list(Path(self.unprocessed_data_path).rglob("*.npy"))
        
        height_paths : list[Path] = []
        meta_paths : list[Path]   = []
        city_paths : list[Path]   = []
        
        for path in all_height_paths:
            
            city_path = path.parent / "cities.csv"
            
            if not city_path.exists():
                continue
            
            meta_path = path.parent / "metadata.json"
            meta_file = json.load(meta_path.open("r"))
            
            if meta_file["nr_cities"] < self.min_cities:
                continue
            
            city_paths.append(city_path)
            height_paths.append(path)
            meta_paths.append(meta_path)
            
        num_train_files = int(len(height_paths) * (1 - self.val_split))
        np.random.seed(self.data_split_seed)
        train_idx = np.random.choice(len(height_paths), num_train_files, replace=False)
        val_idx = np.array([i for i in range(len(height_paths)) if i not in train_idx])
        
        height_train_paths = [height_paths[i] for i in train_idx]
        city_train_paths = [city_paths[i] for i in train_idx]
        
        height_val_paths = [height_paths[i] for i in val_idx]
        city_val_paths = [city_paths[i] for i in val_idx]
        
        return height_train_paths, city_train_paths, height_val_paths, city_val_paths
    

def get_city_dataloader(batch_size=32, shuffle=True, num_workers=4, data_kwargs={}):
    dataset = CityDataset(**data_kwargs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader, dataset


if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    import yaml
    
    with open("CityGeneration/config/config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    dataloader, dataset = get_city_dataloader(data_kwargs=config["dataset"]["params"])
    
    for i in range(len(dataset)):
        
        img, heatmap = dataset[i]
        
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        
        axs[0].imshow(img.permute(1, 2, 0))
        axs[1].imshow(heatmap.permute(1, 2, 0))
        plt.show()
        # dataset._plot_datapoint(i)