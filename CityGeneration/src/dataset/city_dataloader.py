import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json

import pandas as pd
import numpy as np
import tqdm

from torchvision import transforms as T
from PIL import Image

import matplotlib.pyplot as plt

class CityDataset(Dataset):
    
    NORMALIZE_VARIANCE = 3000
    
    def __init__(self,
                 data_path : str|Path,
                 min_cities : int = 2,
                 max_cities : int = 100,
                 img_size : int = 256,
                 cutout_overlap : int = 128,
                 smoothing : int = 20,
                 verbose : bool = True,
                 data_split_seed : int = 42,
                 val_split : float = 0.1,
                 is_train : bool = True):
        
        self.data_path = Path(data_path)
        
        self.min_cities = min_cities
        self.max_cities = max_cities
        self.img_size = img_size
        self.cutout_overlap = cutout_overlap
        
        self.smoothing = smoothing
        
        self.verbose = verbose
        
        self.is_train = is_train
        self.val_split = val_split
        self.data_split_seed = data_split_seed
        
        self.height_paths, self.city_paths = self._extract_city_and_height_paths()
        suffix = "train" if is_train else "val"    
        cache_file = self.data_path / f"citydata_indexing_cache_{suffix}.json"

        use_cache = True
        if cache_file.exists():
            with cache_file.open("r") as f:
                cache_data = json.load(f)
                
                min_cities = cache_data["min_cities"]
                max_cities = cache_data["max_cities"]
                cutout_overlap = cache_data["cutout_overlap"]
                img_size = cache_data["img_size"]
                
                if min_cities != self.min_cities or \
                   max_cities != self.max_cities or \
                   cutout_overlap != self.cutout_overlap or \
                   img_size != self.img_size:
                    
                    print("[STATUS] Cache data does not match current data settings. Recomputing data extraction")
                    use_cache = False
                else:
                    self.extraction_info = cache_data["extraction_info"]
        else:
            use_cache = False
            
            
        if not use_cache:
            self.extraction_info = self._compute_data_extraction()
            
            with cache_file.open("w") as f:
                json.dump({"min_cities" : self.min_cities,
                           "max_cities" : self.max_cities,
                           "img_size" : self.img_size,
                           "cutout_overlap" : self.cutout_overlap,
                           "extraction_info" : self.extraction_info}, f)
        
        if verbose:
            print(f"[DATA INFO, {suffix}] Number of images: ", len(self.height_paths))
            print(f"[DATA INFO, {suffix}] Number of datapoints: ", len(self.extraction_info))
        
    def __len__(self):
        
        assert len(self.height_paths) == len(self.city_paths), "The number of images and cities should be the same"
        
        return len(self.extraction_info)

    def __getitem__(self, idx : int):
        
        i, x, y = self.extraction_info[idx]
        
        img = Image.fromarray(np.load(self.height_paths[i]))
        cities = pd.read_csv(self.city_paths[i])
        
        img, heatmap = self._preprocess(img, cities, x, y)
        
        return img, heatmap
    
    def _preprocess(self, 
                    img : Image,
                    cities : pd.DataFrame,
                    x : int,
                    y : int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        diag_r = int(np.sqrt(2) * (self.img_size / 2))
    
        # Extract 2x diag_r
        img = img.crop((x - diag_r, y - diag_r, x + diag_r, y + diag_r))
        
        cities = cities[(cities["T_x"] > x - diag_r) & (cities["T_x"] < x + diag_r) &
                        (cities["T_y"] > y - diag_r) & (cities["T_y"] < y + diag_r) &
                        (cities["type"] != "administrative")]
        cities = np.array([cities["T_x"] - (x - diag_r), cities["T_y"] - (y - diag_r)]).T
        
        rot_angle = np.random.randint(0, 360)
        
        # Rotate the cities around the center
        rot_matrix = np.array([[np.cos(np.radians(rot_angle)), np.sin(np.radians(rot_angle))],
                               [-np.sin(np.radians(rot_angle)), np.cos(np.radians(rot_angle))]])
        cities = (cities - diag_r) @ rot_matrix.T + diag_r
        
        img = img.rotate(rot_angle)
        
        if np.random.rand() > 0.5:
            cities[:, 0] = 2 * diag_r - 1 - cities[:, 0]
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Extract the img_size x img_size image
        half_w = int(self.img_size / 2)
        img = img.crop((diag_r - half_w, diag_r - half_w, diag_r + half_w, diag_r + half_w))
        cities = cities[(cities[:, 0] > diag_r - half_w) & (cities[:, 0] < diag_r + half_w) &
                        (cities[:, 1] > diag_r - half_w) & (cities[:, 1] < diag_r + half_w)]
        
        # center the cities
        cities = cities - (diag_r - half_w)
        
        img = T.ToTensor()(img)/self.NORMALIZE_VARIANCE
        
        # Apply water mask
        water_mask = img < 0
        img[water_mask] = 0
        
        # city_img = np.zeros((1, self.img_size, self.img_size))
        # city_img[0, cities[:, 1].astype(int), cities[:, 0].astype(int)] = 1
        
        # Convert to heatmap
        heatmap = np.zeros((1, self.img_size, self.img_size))
        
        for city in cities:
            heatmap[0] += np.exp(-((np.arange(self.img_size) - city[0]) ** 2 + 
                                   (np.arange(self.img_size)[:, None] - city[1]) ** 2) /
                                   (2 * self.smoothing ** 2)
                                )
    
        heatmap /= np.sum(heatmap)
        heatmap = torch.tensor(heatmap).float()
        return img, heatmap
    
    def _compute_data_extraction(self):
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
        
        assert self.height_paths, f"No data found at {self.data_path}, call _extract_city_and_height_paths first"
        
        extraction_info = []
        max_city_count = 0
        
        for i, (height_path, city_path) in tqdm.tqdm(enumerate(zip(self.height_paths, self.city_paths)),
                                                     total=len(self.height_paths),
                                                     desc="Extracting data"):
                
            img = np.load(height_path)
            cities = pd.read_csv(city_path)
            
            diag_r = int(np.sqrt(2) * (self.img_size / 2))
            half_w = int(self.img_size / 2)
            
            # Extract the cities from the image
            for x in range(diag_r + half_w, img.shape[0] - diag_r, self.img_size - self.cutout_overlap):
                for y in range(diag_r + half_w, img.shape[1] - diag_r, self.img_size - self.cutout_overlap):
                    
                    city_extract = cities[(cities["T_x"] > x - half_w) & (cities["T_x"] < x + half_w) &
                                          (cities["T_y"] > y - half_w) & (cities["T_y"] < y + half_w) &
                                          (cities["type"] != "administrative")]
                    
                    if len(city_extract) > max_city_count:
                        max_city_count = len(city_extract)
                        
                    if len(city_extract) > self.min_cities and \
                       len(city_extract) < self.max_cities:
                             
                        extraction_info.append((i, x, y))
                        
        if self.verbose:
            print("[DATA INFO] Observed max of cities in a datapoint: ", max_city_count)
                  
        return extraction_info
    
    def _plot_datapoint(self, idx : int):
        
        i, x, y = self.extraction_info[idx]
        
        img = np.load(self.height_paths[i])
        cities = pd.read_csv(self.city_paths[i])
        
        half_w = int(self.img_size / 2)
        
        city_outside = cities[(cities["T_x"] < x - half_w) | (cities["T_x"] > x + half_w) |
                                (cities["T_y"] < y - half_w) | (cities["T_y"] > y + half_w) |
                                (cities["type"] == "hamlet") | (cities["type"] == "administrative")]
        city_extract = cities[(cities["T_x"] > x - half_w) & (cities["T_x"] < x + half_w) &
                              (cities["T_y"] > y - half_w) & (cities["T_y"] < y + half_w) &
                              (cities["type"] != "hamlet") & (cities["type"] != "administrative")]
                        
        plt.imshow(img)
        plt.scatter(city_outside["T_x"],
                    city_outside["T_y"], c="b")
        plt.scatter(city_extract["T_x"],
                    city_extract["T_y"], c="r")
        rect = plt.Rectangle((x - half_w, y - half_w), self.img_size, self.img_size, linewidth=1, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)
        
        for _, row in city_extract.iterrows():
            plt.text(row["T_x"], row["T_y"], row["name"], fontsize=9, ha='right')
        
        plt.show()
        
        
    def _extract_city_and_height_paths(self) -> tuple[list[Path], list[Path]]:
        """
        Extract the paths to the city and height data
        """
        
        # Find all paths that end with ".npy"
        all_height_paths = list(Path(self.data_path).rglob("*.npy"))
        
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
        
        idxs = train_idx if self.is_train else val_idx
            
        height_paths = [height_paths[i] for i in idxs]
        city_paths = [city_paths[i] for i in idxs]
        return height_paths, city_paths
    

def get_city_dataloader(data_path, batch_size=32, shuffle=True, num_workers=4, data_kwargs={}):
    dataset = CityDataset(data_path, **data_kwargs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader, dataset


if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    import yaml
    
    with open("src/city_gen/dataloader_config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    data_path = Path("data/undistorted_data_ortho_3")
    dataloader, dataset = get_city_dataloader(data_path, data_kwargs=config)
    
    for img, city_map, heatmap in dataloader:
        
        batch_idx = 0
        
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        axs[0].imshow(img[batch_idx][0])
        city_idx = city_map[batch_idx].nonzero()
        axs[0].scatter(city_idx[:,2], city_idx[:,1], c="r")
        axs[1].imshow(heatmap[batch_idx][0])

        plt.show()
