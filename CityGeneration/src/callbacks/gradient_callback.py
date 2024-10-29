

import pytorch_lightning as pl

class GradientLoggerCallback(pl.Callback):
    
    def __init__(self, log_every_n, step_type="batch"):
        """
        params:
            log_every_n: int, number of steps between logging
            step_type: str, "batch", "epoch" determines whether to log every n batches or every n epochs
        """
        
        self.log_every_n = log_every_n
        self.step_type = step_type
        
    def _get_gradient_stats(self, model):
        max_grad = 0.0
        mean_grad = 0.0
        for i, param in enumerate(model.parameters()):
            if param.grad is not None:
                grad = param.grad.abs()
                max_grad = max(max_grad, param.grad.abs().max().item())
                mean_grad += (mean_grad - grad.mean().item()) / (i + 1)
        return mean_grad, max_grad
        
    def log(self, trainer, pl_module, batch_idx):
        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                trainer.logger.experiment.add_histogram(f'grad/{name}', param.grad, batch_idx)
                trainer.logger.experiment.add_histogram(f'param/{name}', param, batch_idx)
                    
        mean_grad, max_grad = self._get_gradient_stats(pl_module)
        trainer.logger.experiment.add_scalar('grad/mean', mean_grad, batch_idx)
        trainer.logger.experiment.add_scalar('grad/max', max_grad, batch_idx)
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """ every batch """
        if self.interval == "batch" and batch_idx % self.log_every_n == 0:
            self.log(trainer, pl_module, batch_idx)
    
    def on_train_epoch_end(self, trainer, pl_module):
        """ every epoch """
        if self.interval == "epoch" and trainer.current_epoch % self.log_every_n == 0:
            self.log(trainer, pl_module, trainer.current_epoch)
    
    