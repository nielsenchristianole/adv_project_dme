
import argparse

def dict_to_namespace(d):
    """Recursively converts a dictionary to an argparse.Namespace."""
    if isinstance(d, dict):
        namespace = argparse.Namespace()
        for key, value in d.items():
            setattr(namespace, key, dict_to_namespace(value))
        return namespace
    else:
        return d