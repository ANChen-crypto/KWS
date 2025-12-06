import torch


def dict_to_device(d, device):
    if isinstance(d, torch.Tensor):
        return d.to(device)
    elif isinstance(d, dict):
        return {k: dict_to_device(v, device) for k, v in d.items()}
    elif isinstance(d, list):
        return [dict_to_device(x, device) for x in d]
    elif isinstance(d, tuple):
        return tuple(dict_to_device(x, device) for x in d)
    else:
        return d