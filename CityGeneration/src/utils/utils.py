
import argparse
import importlib
from types import SimpleNamespace as NameSpace
from torch import nn

import copy

def dict_to_namespace(d):
    """Recursively converts a dictionary to an argparse.Namespace."""
    if isinstance(d, dict):
        namespace = argparse.Namespace()
        for key, value in d.items():
            setattr(namespace, key, dict_to_namespace(value))
        return namespace
    else:
        return d
    
def namespace_to_dict(namespace: argparse.Namespace) -> dict:
    if isinstance(namespace, argparse.Namespace):
        # Create a copy to avoid modifying the original Namespace
        d = copy.deepcopy(vars(namespace))
        # Recursively convert inner Namespace objects
        for key, value in d.items():
            d[key] = namespace_to_dict(value)
        return d
    elif isinstance(namespace, list):
        # If there's a list in the Namespace, recurse through each element
        return [namespace_to_dict(item) for item in namespace]
    else:
        return namespace
    
def import_target(target : str) -> any:
    try:
        module_name, class_name = target.rsplit('.', 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except:
        raise ImportError(f"Could not import target: {target}")

def get_model(model_params : NameSpace) -> nn.Module:
    model = import_target(model_params.target)
    return model(**vars(model_params.params))

def get_inference_model(model : nn.Module, wrapped_params : NameSpace) -> nn.Module:
    wrapped = import_target(wrapped_params.target)
    return wrapped(model, **vars(wrapped_params.params))

