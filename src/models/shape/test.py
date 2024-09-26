import torch


from labml_nn.diffusion.stable_diffusion.latent_diffusion import LatentDiffusion, UNetModel, CLIPTextEmbedder
from labml_nn.diffusion.stable_diffusion.sampler.ddpm import DDPMSampler
from labml_nn.diffusion.stable_diffusion.sampler.ddim import DDIMSampler
from labml_nn.diffusion.stable_diffusion.model.autoencoder import Autoencoder, Encoder, Decoder


class DDPMTrainer(DDPMSampler):

    def train_step(self):

        t = torch.randint(0, self.n_steps, (1,)).item()


device = 'cuda' if torch.cuda.is_available() else 'cpu'


data_channels = 1
latent_channels = 2
ae_channel_mults = [1, 2, 4]


unet = UNetModel(
    in_channels=latent_channels,
    out_channels=latent_channels,
    channels=32,
    n_res_blocks=2,
    attention_levels=[1, 2],
    channel_multipliers=[1, 2, 4],
    n_heads=4,
    tf_layers=1,
    d_cond=256)


auto_encoder = Autoencoder(
    Encoder(
        in_channels=data_channels,
        z_channels=latent_channels,
        channels=32,
        n_resnet_blocks=2,
        channel_multipliers=ae_channel_mults),
    Decoder(
        z_channels=latent_channels,
        out_channels=data_channels,
        channels=32,
        n_resnet_blocks=2,
        channel_multipliers=ae_channel_mults[::-1]),
    latent_channels,
    latent_channels)



diff_model = LatentDiffusion(
    unet,
    auto_encoder,
    lambda *args, **kwargs: None,
    latent_scaling_factor=1.,
    n_steps=50,
    linear_start=1e-4,
    linear_end=2e-2)



