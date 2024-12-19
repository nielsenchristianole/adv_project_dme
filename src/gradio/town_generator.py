
from src.gradio.demo_types import Town

from CityGeneration.inference import get_inference_model
from CityGeneration.src.utils.utils import dict_to_namespace
from CityGeneration.src.utils.image_utils import crop_to_square, resize
from CityGeneration.conditional_sampler import ConditionalSampler

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from skimage.morphology import binary_dilation

import yaml
import shapely
import numpy as np

from typing import List

import torch

class TownGenerator:
    def __init__(self, config_path: str, model_config_path: str, checkpoint_path: str):
        
        with open(config_path, 'r') as f:
            config_args = yaml.safe_load(f)

        with open(model_config_path, 'r') as f:
            model_config_args = yaml.safe_load(f)
            
        assert config_args is not None, "Configuration file is empty."
        assert model_config_args is not None, "Model configuration file is empty."

        params = dict_to_namespace(config_args)
        model_params = dict_to_namespace(model_config_args)
        
        self.model = get_inference_model(checkpoint_path, params, model_params)


        kernel = RBF(5, (1e-10, 1e6)) * C(1.0, (1e-10, 1e6))
        self.gp = GaussianProcessRegressor(kernel=kernel, 
                                           optimizer = None,
                                           alpha=1e-6,
                                           n_restarts_optimizer=0)

    def get_masks(self, height_map: np.ndarray, width = 3):
        
        mask = height_map <= 0
        dilated = mask.copy()
        for _ in range(width):
            dilated = binary_dilation(dilated, footprint = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
                
        coastal = mask != dilated
        non_coastal = (1 - mask) ^ coastal
        return mask, coastal, non_coastal

    def generate(self,
                 height_map: np.ndarray,
                 towns: List[Town],
                 names,
                 types,
                 is_coastal,
                 repel = 0.5) -> dict:
        """
        Generates towns on a height map.
        The heightmap is expected to be of shape (128, 128). If not it will automatically be resized.
        
        Args:
         - height_map: np.ndarray, shape=(128, 128)
            The height map to generate towns on.
         - towns: List[Town]
            A list of existing towns
         - names: List[str]
            The names of the new towns
         - types: List[str]
            The types of the new towns
         - is_coastal: List[bool]
            Whether the new towns are coastal or not.
        """
        
        assert len(names) == len(types) == len(is_coastal), "Names, types and is_coastal must have the same length."
        
        original_shape = height_map.shape
        
        assert original_shape[0] == original_shape[1], "Height map must be square."
        
        towns_xyz = np.array([town["xyz"] for town in towns])

        if original_shape != (128, 128):
            height_map = resize(height_map, (128, 128))
            
            if len(towns_xyz) > 0:
                towns_xyz[:, 0] = towns_xyz[:, 0] * 128 / original_shape[0]
                towns_xyz[:, 1] = towns_xyz[:, 1] * 128 / original_shape[1]
        
        mask, coastal, non_coastal = self.get_masks(height_map)
        height_map[mask] = 0
        
        heatmap = self.model(height_map)
        heatmap = heatmap.detach().cpu().numpy().squeeze().squeeze()
        
        new_towns = []
        
        if len(towns_xyz) > 0:
            X_sampled = towns_xyz[:, :2]
            y_sampled = towns_xyz[:, 2].reshape(-1, 1)
        else:
            X_sampled = np.empty((0, 2))
            y_sampled = np.empty((0, 1))
            
        for i in range(len(names)):
            
            if is_coastal[i] == True:
                prior = heatmap * coastal
            else:
                prior = heatmap * non_coastal    
            prior = prior / prior.sum()
            
            self.sampler = ConditionalSampler(self.gp, prior, repel = repel)
            self.sampler.X_sampled = X_sampled
            self.sampler.y_sampled = y_sampled
            
            self.sampler.sample(1)
            
            xyz = np.hstack([self.sampler.X_sampled[-1], self.sampler.y_sampled[-1]])
            
            X_sampled = np.vstack([X_sampled, xyz[:2]])
            y_sampled = np.vstack([y_sampled, xyz[2]])
            
            if original_shape != (128, 128):
                xyz[0] = xyz[0] * original_shape[0] / 128
                xyz[1] = xyz[1] * original_shape[1] / 128
            
            new_towns.append(Town(town_type=types[i],
                            is_coastal=is_coastal[i],
                            xyz=xyz,
                            town_name=names[i],
                            connect=True))
                
        return towns + new_towns
    

if __name__ == "__main__":
    
    import argparse
    from pathlib import Path
    from matplotlib import pyplot as plt
    
    parser = argparse.ArgumentParser(description="Apply inference on a model")
    
    parser.add_argument('--config', type=str,
                        default="out/city_gen/config.yaml",
                        help='Path to the configuration file.')
    parser.add_argument('--model_config', type=str,
                        default="out/city_gen/model_config.yaml",
                        help='Path to the model configuration file.')
    parser.add_argument('--checkpoint', type = str,
                        default = "out/city_gen/checkpoint.ckpt",
                        help="Path to a .ckpt checkpoint file")
    
    args = parser.parse_args()
    
    generator = TownGenerator(args.config,
                              args.model_config,
                              args.checkpoint)
    
    for path in Path("CityGeneration/data/smooth_10").glob("**/*img.npy"):
        img = np.load(path)
        heatmap = np.load(str(path).replace("img","heatmap"))
        
        out = generator.model(img)
        
        img = crop_to_square(img)
        img = resize(img, (128, 128))
        generator.generate(None, img, [], 10)
        
        
        fig, axs = plt.subplots(1,3)
        
        axs[0].imshow(img)
        axs[0].scatter(generator.sampler.X_sampled[:, 0],
                       generator.sampler.X_sampled[:, 1],
                       color='red')
        axs[0].axis('off')
        axs[1].imshow(heatmap)
        axs[2].imshow(out.detach().cpu().numpy().squeeze().squeeze())
        
        plt.show()
        
        