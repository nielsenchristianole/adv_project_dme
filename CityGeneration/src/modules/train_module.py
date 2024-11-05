
from typing import Any

import lightning as L
import torch

from pytorch_optimizer import create_optimizer
from pytorch_lightning.utilities.types import OptimizerLRScheduler

import matplotlib.pyplot as plt

from types import SimpleNamespace as NameSpace
from typing import Union, List

# define the LightningModule
class TrainModule(L.LightningModule):
    
    def __init__(self,
                 model : torch.nn.Module,
                 optimizer_params: Union[NameSpace,None] = None,
                 logging_params: Union[NameSpace,None] = None,
                 loss_functions: Union[List[torch.nn.Module],None] = None,
                 loss_weights: Union[List[float],None] = None):
        super().__init__()
        
        self.model = model
        
        self.loss_functions = loss_functions
        self.loss_weights = loss_weights
        
        self.optimizer_params = optimizer_params
        self.logging_params = logging_params
        
        self.log_idx = 0
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx) -> torch.Tensor:

        data, label = batch

        outputs = self.model(data)
        
        losses = self._get_losses(outputs, label)
        
        self._log_losses(losses, "train")
        self._log_predictions(label, outputs, batch_idx, "train")
            
        return torch.sum(losses)
    
    
    def validation_step(self, batch : tuple, batch_idx) -> torch.Tensor:

        data, label = batch

        outputs = self.model(data)
        
        losses = self._get_losses(outputs, label)
        
        self._log_losses(losses, "val")
        self._log_predictions(label, outputs, batch_idx, "val")

        return torch.sum(losses)
    
    def on_validation_end(self):
        
        return super().on_validation_end()

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
            # use_gc=True,
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
    
    # def on_save_checkpoint(self, checkpoint):#: dict[str, Any]):
    #     super().on_save_checkpoint(checkpoint)
        
    #     checkpoint['name'] =
    
    def _get_losses(self,
                    outputs,# : list[torch.Tensor],
                    batch):# : list[torch.Tensor]) -> list[torch.Tensor]:
        
        losses = torch.zeros(len(self.loss_functions))
        
        for i, (weight, loss_func) in enumerate(zip(self.loss_weights, self.loss_functions)):
            losses[i] = loss_func(outputs, batch) * weight
            
        return losses
        
    def _log_losses(self, losses : torch.Tensor, prefix : str):
        """
        Log the losses to the logger
        """
        on_epoch = True if prefix == "val" else False
        
        for func, loss in zip(self.loss_functions, losses):
            name = func.__class__.__name__
            self.log(f"{prefix}/{name}", loss, on_epoch=on_epoch, on_step = not on_epoch)

        self.log(f"{prefix}/loss", torch.sum(losses), on_epoch=on_epoch, on_step = not on_epoch)

    def _log_predictions(self, labels, outputs, batch_idx, prefix : str):
        """
        Implement logging for predictions
        """
        do_log = False
        if self.logging_params.log_pred.use:
            if prefix == "train":
                if self.logging_params.log_pred.step_type == "batch" and \
                   batch_idx % self.logging_params.log_pred.every_n == 0:
                    do_log = True
                
                if self.logging_params.log_pred.step_type == "epoch" and \
                   self.current_epoch % self.logging_params.log_pred.every_n == 0 and \
                   batch_idx == 0:
                    do_log = True
            elif prefix == "val":
                if batch_idx == 0:
                    do_log = True
                
        if not do_log:  
            return
        
        
        # Logging predictions
        num_plots = min(5, len(outputs))
        fig, axs = plt.subplots(2, num_plots, figsize=(5, 5))
        
        for i in range(num_plots):
            
            # Log predictions
            axs[0, i].imshow(outputs[i].detach().cpu().numpy().transpose(1, 2, 0))
            axs[0, i].axis("off")
        
            # Log labels
            axs[1, i].imshow(labels[i].detach().cpu().numpy().transpose(1, 2, 0))
            axs[1, i].axis("off")
            
        plt.subplots_adjust(wspace=0, hspace=0)
            
        self.logger.experiment.add_figure(f"{prefix}", fig, self.log_idx)
        self.log_idx += 1
        
        plt.close(fig)
        
        return
