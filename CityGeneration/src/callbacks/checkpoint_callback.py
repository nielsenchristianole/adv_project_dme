import lightning as L
from pathlib import Path
from lightning.pytorch.callbacks import Checkpoint

from typing_extensions import override

class EveryNEpochsCallback(Checkpoint):
    def __init__(self, dirpath : str, every_n_epochs : int = 5):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.every_n_epochs = every_n_epochs

    @override
    def on_validation_end(self, trainer : L.Trainer, pl_module : L.LightningModule):
        if (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            trainer.save_checkpoint(self.dirpath / f"epoch-{trainer.current_epoch}.ckpt")
