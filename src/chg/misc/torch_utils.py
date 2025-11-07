import random
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import itertools

from . import utils


def check_cuda():
    if torch.cuda.is_available():
        print("GPU is available!")
        
        # Get the number of GPUs
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")
        
        # Loop through each GPU
        for i in range(num_gpus):
            # Get GPU name
            gpu_name = torch.cuda.get_device_name(i)
            print(f"\nGPU {i}: {gpu_name}")
            
            # Get memory details
            total_memory = torch.cuda.get_device_properties(i).total_memory
            reserved_memory = torch.cuda.memory_reserved(i)
            allocated_memory = torch.cuda.memory_allocated(i)
            free_memory = reserved_memory - allocated_memory
            
            # Print memory details
            print(f"  Total Memory: {total_memory / (1024 ** 3):.2f} GB")
            print(f"  Reserved Memory: {reserved_memory / (1024 ** 3):.2f} GB")
            print(f"  Allocated Memory: {allocated_memory / (1024 ** 3):.2f} GB")
            print(f"  Free Memory: {free_memory / (1024 ** 3):.2f} GB")
    else:
        print("No GPU available.")
        
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def get_memory(tensor, precision=2, unit=None):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Expected a torch.Tensor")
    num_bytes = tensor.numel() * tensor.element_size()
    return utils.format_memory_size(num_bytes, precision=precision, unit=unit)
    
def cross_entropy(input, target, reduction='mean'):
    """
    Wrapper for cross_entropy loss so that the dimensions are more intuitive.
    Permutes the dimensions so that the last dim of input corresponds to number of classes.
    """
    dims = [0, -1] + list(range(1, len(input.shape) - 1))
    input = input.permute(dims)
    return F.cross_entropy(input, target, reduction=reduction)

def logmeanexp(x, dim):
    return x.logsumexp(dim) - torch.log(torch.tensor(x.shape[dim], dtype=x.dtype)).to(x.device)

def to_bytes(vector):
    if isinstance(vector, torch.Tensor):
        vector = vector.cpu().numpy()
    vector = np.asarray(vector, dtype=np.uint8)
    if vector.ndim == 1:
        return np.packbits(vector).tobytes()
    reshaped = vector.reshape(-1, vector.shape[-1])
    packed = np.packbits(reshaped, axis=-1)
    return np.array([row.tobytes() for row in packed], dtype=object).reshape(vector.shape[:-1])

def from_bytes(byte_obj, length, device='cpu'):
    if isinstance(byte_obj, (bytes, bytearray)):
        byte_array = np.frombuffer(byte_obj, dtype=np.uint8)
        bit_array = np.unpackbits(byte_array)[:length]
        return torch.tensor(bit_array, dtype=torch.int, device=device)
    flat = [np.unpackbits(np.frombuffer(b, dtype=np.uint8))[:length] for b in byte_obj.to_numpy()]
    shape = byte_obj.shape
    return torch.tensor(np.stack(flat).reshape(*shape, length), dtype=torch.int, device=device)

def pad_to_same(tensors, pad_value=0, exclude_dims=None):
    """
    Pads a list of tensors to the same shape along all dimensions, except those in exclude_dims.

    Args:
        tensors (list of torch.Tensor): Tensors to pad.
        pad_value (number, optional): Value to use for padding. Default is 0.
        exclude_dims (int or list of int, optional): Dimensions to skip padding (e.g. the concat/stack dim).

    Returns:
        list of torch.Tensor: List of padded tensors, all with identical shapes.
    """
    if exclude_dims is None:
        exclude_dims = []
    elif isinstance(exclude_dims, int):
        exclude_dims = [exclude_dims]
    ndims = tensors[0].ndim
    # Get the target shape for each dimension
    max_shape = list(tensors[0].shape)
    for t in tensors[1:]:
        for d in range(ndims):
            if d not in exclude_dims:
                max_shape[d] = max(max_shape[d], t.shape[d])
    out = []
    for t in tensors:
        shape_diff = [max_shape[d] - t.shape[d] if d not in exclude_dims else 0 for d in range(ndims)]
        pad = []
        for d in reversed(range(ndims)):
            pad.extend([0, int(shape_diff[d])])
        if any(shape_diff):
            t = torch.nn.functional.pad(t, pad, value=pad_value)
        out.append(t)
    return out

def nCk(n, k):
    """
    Compute the number of combinations of `n` items taken `k` at a time.
    """
    n, k = torch.as_tensor(n, dtype=torch.float32), torch.as_tensor(k, dtype=torch.float32)
    return torch.exp(torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1))

def generate_random_binary_vectors(d, k, n, device='cpu'):
    """
    Generates binary vectors of length `d`, where each vector contains exactly `k` ones (1s) 
    and `d - k`. At most `n` unique vectors are generated.

    Args:
        d (int): The length of each binary vector.
        k (int): The number of ones in each binary vector.
        n (int): The maximum number of unique binary vectors to generate.

    Notes:
        - If the number of possible binary vectors (nCr(d, i)) is less than or equal to `k`, all 
          possibilities are included.
        - If `nCr(d, i) > k`, `k` unique vectors are randomly sampled.
    """
    num_possible = round(nCk(d, k).item())
    if num_possible <= n:
        indices = list(itertools.combinations(range(d), k))
    else:
        indices = set()
        while len(indices) < n:
            idx = np.random.choice(d, k, replace=False)
            indices.add(tuple(sorted(idx)))
        indices = list(indices)
    
    indices = torch.tensor(indices, dtype=torch.long, device=device)
    vectors = torch.zeros((len(indices), d), dtype=torch.int, device=device)
    vectors.scatter_(1, indices, 1)
    return vectors

def to_long_df(array, dim_names, value_name: str | list = 'value', **kwargs):
    """
    Converts a multi-dimensional array to a long-format pandas DataFrame with labeled dimensions.

    This function transforms a multi-dimensional `array` (e.g., a numpy array or PyTorch tensor) into a 
    DataFrame in long format, where each row represents a unique combination of indices from each dimension 
    of the array. If `value_name` is a list, the last dimension of `array` is split into separate columns.

    Parameters:
    -----------
    array : numpy.ndarray or torch.Tensor
        A multi-dimensional array containing data to be converted. If it's a torch.Tensor, it will 
        be detached and moved to the CPU.
        
    dim_names : list of str
        Names for each dimension in the array. These will become column names in the DataFrame 
        index for each dimension of `array`.
        
    value_name : str or list of str, default 'value'
        Name(s) for the value columns in the resulting DataFrame. If `value_name` is a single string, 
        all values will be stored under that column name. If `value_name` is a list, each element 
        in the list becomes a column, with each corresponding to the last dimension in `array`. The 
        length of `value_name` must match the size of the last dimension in `array` if a list is provided.

    **kwargs : additional arrays or tensors, optional
        Any additional arrays or tensors to include in the resulting DataFrame as separate columns. 
        These arrays must either match the shape of `array` or be broadcastable to the shape of `array`.
        
    Returns:
    --------
    pandas.DataFrame
        A DataFrame in long format, with one row for each unique combination of indices from the dimensions 
        in `array`. The DataFrame will have a column for each dimension in `dim_names`, and one or more 
        columns for values from `array`, labeled according to `value_name`. Additional columns from `kwargs` 
        are appended if provided.
    
    Notes:
    ------
    - If `array` is a torch.Tensor, it is automatically converted to a numpy array.
    - The `value_name` parameter determines how the data in `array` is represented in the DataFrame:
        - If a single string is provided, the data is stored in a single column with that name.
        - If a list is provided, each element in the list becomes a column name for the respective 
          slice of the last dimension of `array`.
    - Each additional array in `kwargs` is flattened and repeated as necessary to align with the DataFrame.
    
    Example:
    --------
    >>> array = np.random.rand(2, 3, 4)
    >>> dim_names = ['dim1', 'dim2', 'dim3']
    >>> df = to_long_df(array, dim_names, value_name='measurement')
    
    >>> array = np.random.rand(2, 3, 4)
    >>> dim_names = ['dim1', 'dim2']
    >>> value_names = ['feature1', 'feature2', 'feature3', 'feature4']
    >>> df = to_long_df(array, dim_names, value_name=value_names)
    """
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()
    shape = array.shape
    if isinstance(value_name, str):
        array = array.flatten()
        index = pd.MultiIndex.from_product([range(i) for i in shape], names=dim_names)
        df = pd.DataFrame(array, columns=[value_name], index=index).reset_index()
    else:
        array = array.reshape(-1, len(value_name))
        index = pd.MultiIndex.from_product([range(i) for i in shape[:-1]], names=dim_names)
        df = pd.DataFrame(array, columns=value_name, index=index).reset_index()
    for k, v in kwargs.items():
        i = len(v.shape)
        v = v.flatten()
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy()
        while len(v) < len(df):
            v = np.repeat(v, shape[i])
            i += 1
        df[k] = v
    return df