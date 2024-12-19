import os
import sys
from typing import Union, Tuple
from pathlib import Path
from contextlib import contextmanager

import tqdm
import torch
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.models.autoencoder import AutoencoderKL



MODEL_ROOT = Path('./latent-diffusion')


def uniform_to_height(u: torch.Tensor, *, mean: float=300.0) -> torch.Tensor:
    """
    Transform uniform samples to height samples
    u ~ U(0, 1)
    height ~ Exp(mean)
    """
    return - mean * torch.log(1 - u)


def load_models(model_configs: dict, *, device: str='cuda', model_root: Path=MODEL_ROOT) -> Tuple[LatentDiffusion, LatentDiffusion]:

    def load_model(model_name: str) -> Union[AutoencoderKL, LatentDiffusion]:
        time, model = model_configs[model_name]['exp_name'].split('_')
        epoch = model_configs[model_name]['epoch']
        log_dir = Path(f"logs/{time}_{model}")


        config_path = model_root / log_dir / f'configs/{time}-project.yaml'
        assert config_path.exists(), f"Config file not found: {config_path}"

        config = OmegaConf.load(config_path)
        config['model']['params']['ckpt_path'] = str(
            log_dir / f'checkpoints/epoch={epoch:>06}.ckpt')

        current_dir = os.getcwd()
        os.chdir(model_root)
        model: LatentDiffusion = instantiate_from_config(config['model'])
        os.chdir(current_dir)
        return model.to(device).eval()


    shape_model: LatentDiffusion = load_model('shape_ldm')
    shape_model.first_stage_model.negative_1_to_1 = False
    shape_model.first_stage_model.im_recon_mode = 'continuous_binary'

    height_model: LatentDiffusion = load_model('height_ldm')
    height_model.first_stage_model.negative_1_to_1 = False
    height_model.first_stage_model.im_recon_mode = 'continuous'

    # check if shape and height models are compatible
    for p1, p2 in zip(shape_model.first_stage_model.parameters(), height_model.cond_stage_model.parameters()):
        if p1.data.ne(p2.data).any() > 0:
            raise ValueError('Shape first stage and height cond stage different')
        
    return shape_model, height_model


@contextmanager
def suppress_logs():
    with open(os.devnull, 'w') as devnull:
        stdout = sys.stdout
        stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:  
            yield
        finally:
            sys.stdout = stdout
            sys.stderr = stderr