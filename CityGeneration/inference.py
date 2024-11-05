
import argparse
import yaml

from CityGeneration.src.utils.utils import dict_to_namespace, get_model, get_wrapped_model
from CityGeneration.src.modules.train_module import TrainModule

from types import SimpleNamespace as NameSpace


def get_inference_model(ckpt_path : str,
                        params : NameSpace,
                        model_params : NameSpace) -> None:
    
    model = get_model(model_params)
    model = TrainModule.load_from_checkpoint(ckpt_path, model = model)
    
    wrapped = get_wrapped_model(model, params.inference)
    
    return wrapped

if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(description="Apply inference on a model")
    
    parser.add_argument('--config', type=str,
                        default="CityGeneration/out/logs/train-1105/version_35/config.yaml",
                        help='Path to the configuration file.')
    parser.add_argument('--model_config', type=str,
                        default="CityGeneration/out/logs/train-1105/version_35/model_config.yaml",
                        help='Path to the model configuration file.')
    parser.add_argument('--checkpoint', type = str,
                        default = "CityGeneration/out/logs/train-1105/version_35/epoch-7.ckpt",
                        help="Path to a .ckpt checkpoint file")
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config_args = yaml.safe_load(f)

    with open(args.model_config, 'r') as f:
        model_config_args = yaml.safe_load(f)
        
    assert config_args is not None, "Configuration file is empty."
    assert model_config_args is not None, "Model configuration file is empty."

    params = dict_to_namespace(config_args)
    model_params = dict_to_namespace(model_config_args)
    
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
        
        