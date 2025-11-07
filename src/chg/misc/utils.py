import yaml
import os
import socket
import torch
import numpy as np
from pathlib import Path


def get_directories(config_filename: str) -> dict:
    with open(config_filename, "r") as f:
        config = yaml.safe_load(f)

    hostname = socket.gethostname()
    cluster = "tiger" if "tiger" in hostname else "della" if "della" in hostname else None
    user = os.getenv("USER") or os.getenv("LOGNAME")

    if cluster is None or user is None:
        raise RuntimeError("Failed to detect cluster or user.")

    cluster_dirs = config["directories"]["cluster"].get(cluster, {})
    user_dirs = config["directories"]["user"].get(user, {})

    return {k: Path(v) for k, v in {**cluster_dirs, **user_dirs}.items()}


def kv_print(sep=" | ", digits=2, sci=False, **kwargs):
    fmt = f"{{:.{digits}{'e' if sci else 'f'}}}"
    def format_val(v):
        if isinstance(v, (float, np.floating)):
            return fmt.format(v)
        if isinstance(v, (int, np.integer)):
            return str(v)
        if torch.is_tensor(v) and v.numel() == 1:
            v_cpu = v.cpu().item()
            return fmt.format(v_cpu) if isinstance(v_cpu, float) else str(v_cpu)
        if isinstance(v, np.ndarray) and v.size == 1:
            v_item = v.item()
            return fmt.format(v_item) if isinstance(v_item, float) else str(v_item)
        return str(v)
    out = sep.join([f"{k}: {format_val(v)}" for k, v in kwargs.items()])
    print(out)


def format_memory_size(num_bytes, precision=2, unit=None):
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    if unit is None:
        scale = min(len(units) - 1, int(num_bytes).bit_length() // 10)
        unit = units[scale]
    else:
        unit = unit.upper()
        if unit not in units:
            raise ValueError(f"Invalid unit '{unit}', must be one of {units}")
        scale = units.index(unit)

    scaled = num_bytes / (1024 ** scale)
    return f"{scaled:.{precision}f} {unit}"


def flatten(lst):
    """
    Recursively flattens any nested combination of lists or tuples into a flat list.
    """
    return [item for sublist in lst for item in (flatten(sublist) if isinstance(sublist, (list, tuple)) else [sublist])]