
from src.gradio.demo_types import Town

from CityGeneration.inference import get_inference_model
from CityGeneration.src.utils.utils import dict_to_namespace
from CityGeneration.src.utils.image_utils import crop_to_square, resize
from CityGeneration.conditional_sampler import ConditionalSampler

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

import yaml
import shapely
import numpy as np

from typing import List

class TownGenerator:
    def __init__(self,
                 config_path : str,
                 model_config_path : str,
                 checkpoint_path : str):
        
        with open(config_path, 'r') as f:
            config_args = yaml.safe_load(f)

        with open(model_config_path, 'r') as f:
            model_config_args = yaml.safe_load(f)
            
        assert config_args is not None, "Configuration file is empty."
        assert model_config_args is not None, "Model configuration file is empty."

        params = dict_to_namespace(config_args)
        model_params = dict_to_namespace(model_config_args)
        
        self.model = get_inference_model(checkpoint_path, params, model_params)

        kernel = RBF(5, (1e-10, 1e6))
        self.gp = GaussianProcessRegressor(kernel=kernel, 
                                      optimizer = None,
                                      alpha=1e-6, n_restarts_optimizer=0)

    def generate(self,
                 shape: shapely.MultiPolygon,
                 height_map: np.ndarray,
                 towns: List[Town],
                 num_towns: int) -> dict:
        
        heatmap = self.model(height_map)
        heatmap = heatmap.detach().cpu().numpy().squeeze().squeeze()
        heatmap[height_map == 0] = 0
        
        self.sampler = ConditionalSampler(self.gp, 
                                          heatmap,
                                          repel = 0.5)
        
        self.sampler.reset()
        for town in towns:
            self.sampler.X_sampled = np.vstack([self.sampler.X_sampled, town.xyz[:2]])
            self.sampler.y_sampled = np.vstack([self.sampler.y_sampled, town.xyz[2]])
        
        self.sampler.sample(num_towns)
    

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
        
        