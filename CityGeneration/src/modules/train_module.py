
from typing import Any

import lightning as L
import torch

from pytorch_optimizer import create_optimizer
from pytorch_lightning.utilities.types import OptimizerLRScheduler

import matplotlib.pyplot as plt

# define the LightningModule
class TrainModule(L.LightningModule):
    
    def __init__(self,
                 model : torch.nn.Module,
                 optimizer_params,
                 loss_functions : list[torch.nn.Module],
                 loss_weights : list[float]|None):
        super().__init__()
        
        self.model = model
        
        assert len(loss_functions) == len(loss_weights), "Loss functions and weights must be the same length"
        
        self.loss_functions = loss_functions
        self.loss_weights = loss_weights
        
        self.optimizer_params = optimizer_params
        
        self.save_hyperparameters(ignore=['model'])
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx) -> torch.Tensor:

        data, label = batch

        outputs = self.model(data)
        
        losses = self._get_losses(outputs, label)
        self._log_losses(losses, "train")
        
        if batch_idx == 0:
            self._log_predictions(label, outputs, "train")
            
        return torch.sum(losses)
    
    
    def validation_step(self, batch : tuple, batch_idx) -> torch.Tensor:

        data, label = batch

        outputs = self.model(data)
        
        losses = self._get_losses(outputs, label)
        
        if batch_idx == 0:
            self._log_predictions(label, outputs, "val")

        return torch.sum(losses)

    def _get_sarphiv_schedular(self, optimizer):
        """
        Sequential Annealing Restarting Periodical Hybrid Initialized Velocity schedular
        """

        # NOTE: Must instantiate cosine scheduler first,
        #  because super scheduler mutates the initial learning rate.
        lr_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=int(self.optimizer_params.lr_half_period),
            T_mult=int(self.optimizer_params.lr_mult_period),
            eta_min=float(self.optimizer_params.lr_min)
        )
        lr_super = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=float(self.optimizer_params.lr_warmup_max),
            total_steps=int(self.optimizer_params.lr_warmup_period),
        )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer=optimizer,
            schedulers=[lr_super, lr_cosine], # type: ignore
            milestones=[int(self.optimizer_params.lr_warmup_period)],
        )
        
        return lr_scheduler

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = create_optimizer(
            self.model, # type: ignore
            self.optimizer_params.optimizer,
            lr=float(self.optimizer_params.lr_max),
            weight_decay=float(self.optimizer_params.weight_decay),
            use_lookahead=True,
            use_gc=True,
            eps=1e-6
        )

        lr_scheduler = self._get_sarphiv_schedular(optimizer)

        return { # type: ignore
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "lr"
            }
        }
    
    def on_save_checkpoint(self, checkpoint: dict[str, Any]):
        super().on_save_checkpoint(checkpoint)
        
        checkpoint['name'] = self.model.NAME
    
    def _get_losses(self, outputs : list[torch.Tensor], batch : list[torch.Tensor]) -> list[torch.Tensor]:
        
        losses = torch.zeros(len(self.loss_functions))
        
        for i, (weight, loss_func) in enumerate(zip(self.loss_weights, self.loss_functions)):
            losses[i] = loss_func(outputs, batch) * weight
            
        return losses
        
    def _log_losses(self, losses : torch.Tensor, subdir : str):
        """
        Log the losses to the logger
        """
        for func, loss in zip(self.loss_functions, losses):
            name = func.__class__.__name__
            self.log(f"{subdir}/{name}", loss, on_epoch=True, on_step=False)

    def _log_predictions(self, batch, outputs, prefix : str):
        """
        Implement logging for predictions
        """
        
        labels = batch[1]
        
        fig, axs = plt.subplots(1, 5, figsize=(10, 5))
        
        for i, ax in enumerate(axs):
            ax.imshow(outputs[i].detach().cpu().numpy().transpose(1, 2, 0))
            ax.set_title(f"Prediction {i}")
            ax.axis("off")
        
        self.logger.experiment.add_figure(f"{prefix}/predictions", fig)
        
        plt.close(fig)
        
        fig, axs = plt.subplots(1, 5, figsize=(10, 5))
        
        for i, ax in enumerate(axs):
            ax.imshow(labels[i].detach().cpu().numpy().transpose(1, 2, 0))
            ax.set_title(f"Label {i}")
            ax.axis("off")
            
        self.logger.experiment.add_figure(f"{prefix}/labels", fig)
        
        plt.close(fig)
        
        return
