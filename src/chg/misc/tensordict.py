import torch
from collections.abc import Mapping
from collections import OrderedDict
import torch.nn.functional as F
import numpy as np
from collections import namedtuple

from . import utils, torch_utils as tu


class TensorDict(Mapping):
    """
    A dictionary-like container for torch.Tensors with enhanced structure, device consistency,
    and batch-level operations.

    Compared to a plain dict of tensors, TensorDict:
      - Enforces all tensors share the same device.
      - Supports attribute-style access (e.g., td.x instead of td['x']).
      - Propagates tensor operations (e.g., td.cuda(), td.mean()) to all fields.
      - Supports nested TensorDicts and recursive transformations (e.g., td.map(fn)).
      - Includes utilities for memory usage, partitioning, stacking, and printing.

    Ideal for organizing minibatches, model outputs, and structured data.

    Example:
        td = TensorDict(obs=torch.randn(32, 4), act=torch.randint(0, 3, (32,)))
        td = td.cuda().map(lambda x: x * 2)
    """
    
    def __init__(self, **kwargs):
        self._dict = OrderedDict()
        devices = {v.device for v in kwargs.values() if isinstance(v, torch.Tensor)}
        if len(devices) > 1:
            raise ValueError(f"All tensors must be on the same device. Devices found: {devices}")
        for k, v in kwargs.items():
            self[k] = v  # uses __setitem__
            
    @property
    def device(self):
        if self._dict:
            for v in self._dict.values():
                if isinstance(v, torch.Tensor):
                    return v.device
                elif isinstance(v, TensorDict):
                    return v.device
        return None
            
    @staticmethod
    def cat(tds, dim=0, pad_value=None):
        """
        Concatenates a list of TensorDicts along the specified dimension.
        If pad_value is set, all tensors are padded (except along the concat dimension) to match the maximum shape for each key.
        Recursively handles nested TensorDicts.
        """
        keys = tds[0].keys()
        new_td = {}
        for k in keys:
            values = [td[k] for td in tds]
            if isinstance(values[0], torch.Tensor):
                tensors = values
                if pad_value is not None:
                    tensors = tu.pad_to_same(tensors, pad_value=pad_value, exclude_dims=dim)
                new_td[k] = torch.cat(tensors, dim=dim)
            else:
                new_td[k] = TensorDict.cat(values, dim=dim, pad_value=pad_value)
        return TensorDict(**new_td)

    @staticmethod
    def stack(tds, dim=0, pad_value=None):
        """
        Stacks a list of TensorDicts along a new dimension.
        If pad_value is set, all tensors are padded to match the maximum shape for each key along all dimensions.
        Recursively handles nested TensorDicts.
        """
        keys = tds[0].keys()
        new_td = {}
        for k in keys:
            values = [td[k] for td in tds]
            if isinstance(values[0], torch.Tensor):
                tensors = values
                if pad_value is not None:
                    tensors = tu.pad_to_same(tensors, pad_value=pad_value)
                new_td[k] = torch.stack(tensors, dim=dim)
            else:
                new_td[k] = TensorDict.stack(values, dim=dim, pad_value=pad_value)
        return TensorDict(**new_td)

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._dict[key]
        if isinstance(key, (list, tuple)) and isinstance(key[0], str):
            return TensorDict(**{k: self._dict[k] for k in key})
        return TensorDict(**{k: v[key] for k, v in self._dict.items()})

    def __setitem__(self, key, value):
        if isinstance(value, (torch.Tensor, TensorDict)):
            if self._dict:
                self.check_device(value)
            self._dict[key] = value
        elif isinstance(value, Mapping):
            td_value = TensorDict(**value)
            if self._dict:
                self.check_device(td_value)
            self._dict[key] = td_value
        else:
            device = self.device or 'cpu'
            self._dict[key] = torch.tensor(value, device=device)

    def __getattr__(self, item):
        if item in self._dict:
            return self._dict[item]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")

    def __setattr__(self, key, value):
        if '_dict' in self.__dict__ and key in self._dict:
            if not isinstance(value, (torch.Tensor, TensorDict)):
                raise TypeError(f"Cannot assign non-tensor to registered tensor field '{key}'")
            self.check_device(value)
            self._dict[key] = value
        else:
            super().__setattr__(key, value)

    def __delitem__(self, key):
        del self._dict[key]

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)
    
    def __contains__(self, key):
        return key in self._dict
    
    def __repr__(self):
        def fmt(k, v):
            if isinstance(v, torch.Tensor):
                shape = f"({v.shape[0]})" if v.ndim == 1 else f"{tuple(v.shape)}"
                dtype = str(v.dtype).replace("torch.", "")
                return f"{k}[{dtype}]: {shape}"
            elif isinstance(v, TensorDict):
                return f"{k}: {v.__class__.__name__}"
            else:
                return f"{k}: {type(v).__name__}"
        keys = ', '.join(fmt(k, v) for k, v in self._dict.items())
        mem = self.memory_bytes_str()
        return f"{self.__class__.__name__}[{mem}] ({keys})"

    def check_device(self, item):
        if hasattr(item, 'device') and self.device is not None and item.device != self.device:
            raise ValueError(f"Tried to set Tensor on device {item.device} to TensorDict on device {self.device}.")
        
    def update(self, other):
        if not isinstance(other, (Mapping, TensorDict)):
            raise TypeError("update() expects a Mapping or TensorDict")
        for k, v in other.items():
            self[k] = v  # __setitem__ enforces device consistency

    def map(self, f):
        tensors = {}
        for k, v in self._dict.items():
            if isinstance(v, TensorDict):
                tensors[k] = v.map(f)
            elif isinstance(v, torch.Tensor):
                tensors[k] = f(v)
            else:
                raise TypeError(f"Cannot apply map to non-tensor field '{k}' of type {type(v)}")
        return self.__class__(**tensors)
    
    def save(self, path):
        """
        Save the TensorDict to disk as a torch file (.pt or .pth).

        Args:
            path (str): File path to save the TensorDict.
        """
        torch.save(self.to_dict(), path)

    @classmethod
    def load(cls, path, map_location=None):
        """
        Load a TensorDict from disk.

        Args:
            path (str): File path to load from.
            map_location (optional): Device mapping for torch.load.

        Returns:
            TensorDict: Loaded instance.
        """
        state = torch.load(path, map_location=map_location)
        return cls(**state)

    def to_dict(self):
        """
        Recursively convert to a standard Python dict for saving.
        """
        out = {}
        for k, v in self._dict.items():
            if isinstance(v, TensorDict):
                out[k] = v.to_dict()
            else:
                out[k] = v
        return out
    
    def numpy(self):
        def convert(d):
            fields = {}
            for k, v in d._dict.items():
                if isinstance(v, TensorDict):
                    fields[k] = convert(v)
                elif isinstance(v, torch.Tensor):
                    fields[k] = v.detach().cpu().numpy()
                else:
                    raise TypeError(f"Unsupported type: {type(v)} for key {k}")
            NT = namedtuple('TensorDictTuple', fields.keys())
            return NT(**fields)
        return convert(self)
    
    def partition(self, parts, seed=0, drop_remainder=True):
        """
        Partition the dataset into multiple splits based on counts or proportions.
        Pads to full dataset length, then trims final splits if drop_remainder is True.

        Args:
            parts (List[int] or List[float]): List of partition sizes. Must be all counts or all proportions.
            seed (int): Seed for shuffling.
            drop_remainder (bool): If True, drop any samples beyond the sum of `parts`.

        Returns:
            List[self.__class__]: List of dataset partitions.

        Raises:
            ValueError: If parts mix types or total exceeds dataset size.
        """
        lengths = [v.shape[0] for v in self._dict.values()]
        if len(set(lengths)) != 1:
            raise ValueError("All entries in TensorDict must have the same length along dim=0")

        N = len(next(iter(self._dict.values())))
        is_proportions = all(isinstance(p, float) for p in parts)
        is_counts = all(isinstance(p, int) for p in parts)
        if not (is_proportions or is_counts):
            raise ValueError("Parts must be all floats or all ints")
        counts = [int(N * p) for p in parts] if is_proportions else parts
        if sum(counts) > N:
            raise ValueError("Requested partition sizes exceed dataset length")
        generator = torch.Generator(device=self.device).manual_seed(seed)
        idx = torch.randperm(N, generator=generator, device=self.device)

        splits, cursor = [], 0
        for count in counts:
            sel = idx[cursor:cursor + count]
            subdict = {k: v[sel] for k, v in self._dict.items()}
            splits.append(TensorDict(**subdict))
            cursor += count
        if not drop_remainder and cursor < N:
            sel = idx[cursor:]
            subdict = {k: v[sel] for k, v in self._dict.items()}
            splits.append(TensorDict(**subdict))
        return splits
    
    def memory_bytes(self):
        return sum(v.memory_bytes() if isinstance(v, TensorDict)
                else v.numel() * v.element_size()
                for v in self._dict.values() if isinstance(v, (TensorDict, torch.Tensor)))

    def memory_bytes_str(self, precision=2, unit=None):
        return utils.format_memory_size(self.memory_bytes(), precision=precision, unit=unit)
    
    def print(self, round_digits=2, detach=True, cpu=True, print_str=True, return_str=False, max_lines=3):
        td = self
        if detach:
            td = td.detach()
        if cpu:
            td = td.cpu()
        lines = [f"{self.__class__.__name__}[{self.memory_bytes_str()} on {self.device}] with {len(self)} keys"]
        for k, v in td._dict.items():
            if isinstance(v, TensorDict):
                sub = v.print(round_digits=round_digits, detach=False, cpu=False, print_str=False, return_str=True, max_lines=max_lines)
                lines.append(f"{k}: " + "\n  ".join(sub.splitlines()))
            elif isinstance(v, torch.Tensor):
                t = torch.round(v * 10**round_digits) / 10**round_digits if v.dtype.is_floating_point else v
                formatted_lines = repr(t).splitlines()
                preview = formatted_lines[:max_lines]
                if len(formatted_lines) > max_lines:
                    indent = formatted_lines[-1][:len(formatted_lines[-1]) - len(formatted_lines[-1].lstrip())]
                    preview.append(f"{indent}...")
                preview = "\n  ".join(preview)
                lines.append(f"{k}: {str(v.dtype)[6:]} tensor with shape {tuple(v.shape)}\n  {preview}")
            else:
                lines.append(f"{k}: {v}")
        out = "\n".join(lines)
        if print_str:
            print(out)
        if return_str:
            return out

methods = [
    'detach', 'cpu', 'cuda', 'float', 'double', 'half', 'int', 'long',
    'bfloat16', 'short', 'byte', 'bool', 'clone', 'contiguous',
    'squeeze', 'unsqueeze', 'mean', 'sum', 'softmax', 'log_softmax',
    'transpose', 'permute', 'abs', 'sqrt', 'tanh', 'relu', 'to'
]

for name in methods:
    def method(self, *args, name=name, **kwargs):
        return self.map(lambda t: getattr(t, name)(*args, **kwargs))
    setattr(TensorDict, name, method)

