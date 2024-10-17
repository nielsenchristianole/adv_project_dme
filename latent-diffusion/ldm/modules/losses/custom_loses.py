from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import ContinuousBernoulli

from taming.modules.losses.vqperceptual import LPIPS, NLayerDiscriminator, weights_init, hinge_d_loss,  vanilla_d_loss, adopt_weight
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution


class STEFunction(torch.autograd.Function):
    """
    We use this at the discriminator takes binarized images, but we want to train with the gradients

    For the backward pass we clamp the gradients to -1, 1.
    """
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float() * 2.0 - 1.0

    @staticmethod
    def backward(ctx, grad_output):
        return torch.nn.functional.hardtanh(grad_output)


class VaeLossWithDiscriminator(ABC, nn.Module):
    """
    Make sure that the input is not binarized
    """
    def __init__(self, disc_start, kl_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False,
                 disc_loss="hinge"):
        
        """
        disc_start:
            Number of warmup steps without training the discriminator
        kl_weight:
            Weight of the KL loss
        pixelloss_weight:
            Weight of the pixel loss
        disc_num_layers:
            Number of layers in the discriminator
        disc_in_channels:
            Number of input channels for the discriminator
        disc_factor:
            Factor for the discriminator loss
        perceptual_weight:
            Weight of the perceptual loss
        use_actnorm:
            Use BatchNorm2d in the discriminator
        disc_loss:
            Type of GAN loss
        """

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight
    
    @abstractmethod
    def _forward(self, inputs: torch.Tensor, reconstructions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This function takes the inputs and reconstructions and returns the nll loss and a processed reconstruction to be handed to the discriminator and perceptual loss
        """

    def forward(self, inputs: torch.Tensor, reconstructions: torch.Tensor, posteriors: DiagonalGaussianDistribution, optimizer_idx: int,
                global_step: int, last_layer: torch.Tensor, split: str="train"):
        
        nll_loss, processed_reconstructions = self._forward(inputs, reconstructions)

        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), processed_reconstructions.contiguous())
            nll_loss = nll_loss + self.perceptual_weight * p_loss
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]

        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        # now the GAN part
        if optimizer_idx == 0:
            logits_fake = self.discriminator(processed_reconstructions.contiguous())
            g_loss = -torch.mean(logits_fake)

            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/kl_loss".format(split): kl_loss.detach().mean(), "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): nll_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            logits_real = self.discriminator(inputs.contiguous().detach())
            logits_fake = self.discriminator(processed_reconstructions.contiguous().detach())

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log


class BernoulliWithDiscriminator(VaeLossWithDiscriminator):

    def _forward(self, inputs: torch.Tensor, reconstructions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        nll_loss = torch.nn.functional.binary_cross_entropy_with_logits(reconstructions.contiguous(), inputs.contiguous(), reduction='none')
        binarized_reconstructions = STEFunction.apply(reconstructions.contiguous())

        return nll_loss, binarized_reconstructions


class ContinuousBernoulliWithDiscriminator(VaeLossWithDiscriminator):

    def _forward(self, inputs: torch.Tensor, reconstructions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        dist = ContinuousBernoulli(logits=reconstructions.contiguous())
        return dist.log_prob(inputs.contiguous()), dist.probs.contiguous()
