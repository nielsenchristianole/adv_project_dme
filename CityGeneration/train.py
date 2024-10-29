
import datetime
import argparse

import lightning as L
from torch import nn
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from src.modules.train_module import TrainModule
from src.modules.data_module import DataModule
from src.utils.misc import dict_to_namespace
from src.callbacks.gradient_callback import GradientLoggerCallback

import yaml
import importlib
from pathlib import Path

from types import SimpleNamespace as NameSpace

def import_target(target : str) -> any:
    try:
        module_name, class_name = target.rsplit('.', 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except:
        raise ImportError(f"Could not import target: {target}")
    
def get_loss_functions(params : NameSpace) -> tuple[list[nn.Module], list[float]]:

    weights = []
    functions = []
    for info in  params.training.loss_functions:
        loss = import_target(info["target"])
        functions.append(loss(**info["params"]))
        weights.append(info["weight"])
        
    return functions, weights

def get_model(model_params : NameSpace) -> nn.Module:
    model = import_target(model_params.target)
    return model(**vars(model_params.params))

def save_config(params : NameSpace, path : str) -> None:
    with open(path, 'w') as f:
        yaml.dump(vars(params), f)

def train(params : NameSpace, model_params : NameSpace) -> None:
    
    # Lightning seed everything
    L.seed_everything(params.seed)
    
    # Define the dataset and the data module
    dataset = import_target(params.dataset.target)
    data_module = DataModule(dataset = dataset,
                             dataset_params = params.dataset.params,
                             dataloader_params = params.dataloader)
    print("[STATUS] Data module created without errors.")
    
    # Define the loss function, model, and training module
    loss_functions, loss_weights = get_loss_functions(params)
    model = get_model(model_params)
    train_module = TrainModule(model = model,
                               loss_functions = loss_functions,
                               loss_weights = loss_weights,
                               optimizer_params = params.training.optimizer)
    print("[STATUS] Training module created without errors.")
    
    suffix = datetime.datetime.now().strftime(params.identifier.time_format)
    identifier = f"{params.identifier.name}-{suffix}"
    
    # Logger
    logger = TensorBoardLogger(Path(params.logging.dir),
                               name=identifier)
    
    save_config(params, logger.log_dir / "config.yaml")
    save_config(model_params, logger.log_dir / "model_config.yaml")
    
    # Define the callbacks
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        monitor='val/loss', 
        dirpath=params.checkpointing.dir,
        filename=identifier,
        save_top_k=params.checkpointing.save_top_k,
        mode=params.checkpointing.mode,
    )
    
    callbacks = [lr_monitor, checkpoint_callback]
    
    if params.logging.log_gradients.use:
        gradient_callback = GradientLoggerCallback(log_every_n=params.logging.log_gradients.every_n,
                                                   step_type=params.logging.log_gradients.step_type)
        callbacks.append(gradient_callback)
    
    # Gradient clipping
    if params.training.gradient_clipping.use:
        gradient_clip_val = params.training.gradient_clipping.value
        gradient_clip_algorithm = params.training.gradient_clipping.algorithm
    else:
        gradient_clip_val = None
        gradient_clip_algorithm = None
    
    trainer = L.Trainer(accelerator=params.training.accelerator,
                        callbacks=callbacks,
                        max_epochs=params.training.epochs,
                        log_every_n_steps=params.logging.log_every_n_steps,
                        accumulate_grad_batches=params.training.accumulate_grad_batches,
                        gradient_clip_val=gradient_clip_val,
                        gradient_clip_algorithm=gradient_clip_algorithm,
                        logger=logger)
    print("[STATUS] Trainer created without errors.")
    
    trainer.fit(train_module, datamodule=data_module)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train the resnet model on the ear dataset.")
    parser.add_argument('--config', type=str,
                        default="CityGeneration/config/config.yaml",
                        help='Path to the configuration file.')
    parser.add_argument('--model_config', type=str,
                        default="CityGeneration/config/model_config.yaml",
                        help='Path to the model configuration file.')
    
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config_args = yaml.safe_load(f)

    with open(args.model_config, 'r') as f:
        model_config_args = yaml.safe_load(f)
        
    assert config_args is not None, "Configuration file is empty."
    assert model_config_args is not None, "Model configuration file is empty."

    params = dict_to_namespace(config_args)
    model_params = dict_to_namespace(model_config_args)

    train(params, model_params)