
import argparse
import yaml

from CityGeneration.src.utils import utils
#from CityGeneration.src.modules.train_module import TrainModule
# from pytorch_lightning.core.lightning import LightningModule
from types import SimpleNamespace as NameSpace

import torch

def get_inference_model(ckpt_path : str,
                        params : NameSpace,
                        model_params : NameSpace) -> None:
    
    ckpt_data = torch.load(ckpt_path, map_location=torch.device('cpu'))
    model = utils.get_model(model_params)
    new_state_dict = {k.replace("model.", ""): v for k, v in ckpt_data["state_dict"].items()}
    model.load_state_dict(new_state_dict)
    wrapped = utils.get_inference_model(model, params.inference)
    
    return wrapped

if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(description="Apply inference on a model")
    
    parser.add_argument('--config', type=str,
                        default="out\city_gen\config.yaml",
                        help='Path to the configuration file.')
    parser.add_argument('--model_config', type=str,
                        default="out\city_gen\model_config.yaml",
                        help='Path to the model configuration file.')
    parser.add_argument('--checkpoint', type = str,
                        default = "out\city_gen\checkpoint.ckpt",
                        help="Path to a .ckpt checkpoint file")
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config_args = yaml.safe_load(f)

    with open(args.model_config, 'r') as f:
        model_config_args = yaml.safe_load(f)
        
    assert config_args is not None, "Configuration file is empty."
    assert model_config_args is not None, "Model configuration file is empty."

    params = utils.dict_to_namespace(config_args)
    model_params = utils.dict_to_namespace(model_config_args)
    
    model = get_inference_model(args.checkpoint, params, model_params)
    
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    for path in Path("CityGeneration/data/smooth_10").glob("**/*img.npy"):
        img = np.load(path)
        heatmap = np.load(str(path).replace("img","heatmap"))
        
        out = model(img)
        
        fig, axs = plt.subplots(1,3)
        
        axs[0].imshow(img)
        axs[1].imshow(heatmap)
        axs[2].imshow(out.detach().cpu().numpy().squeeze().squeeze())
        
        plt.show()
        
        