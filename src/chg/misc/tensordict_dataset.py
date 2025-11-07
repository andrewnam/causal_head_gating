import torch
from torch.utils.data import Dataset, DataLoader
from .tensordict import TensorDict


import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from .tensordict import TensorDict


class TensorDictDataset(TensorDict):
    """
    PyTorch Dataset wrapper for TensorDicts, with batch collation and DataLoader helpers.

    Args:
        **kwargs: Named tensors with equal first dimension (e.g., text_tokens=..., labels=...).

    Methods:
        collate_fn(batch): Stacks a list of TensorDicts into a batch.
        to_dataloader(infinite=False, **kwargs): Returns a DataLoader or infinite loader.
        to_iterator(**kwargs): Infinite iterator over the DataLoader.

    Example:
        dataset = TensorDictDataset(text_tokens=..., labels=...)
        for batch in dataset.to_dataloader(batch_size=8):
            print(batch["text_tokens"].shape)
    """
    def __len__(self):
        return len(next(iter(self._dict.values())))

    def __getitem__(self, index):
        return TensorDict(**{k: v[index] for k, v in self._dict.items()})

    def collate_fn(self, batch):
        return TensorDict.stack(batch)

    def to_dataloader(self, infinite=False, **kwargs):
        def infinite_loader():
            while True:
                yield from DataLoader(self, collate_fn=self.collate_fn, **kwargs)
        return infinite_loader() if infinite else DataLoader(self, collate_fn=self.collate_fn, **kwargs)

    def to_iterator(self, **kwargs):
        return iter(self.to_dataloader(infinite=True, **kwargs))


class MaskedSequenceDataset(TensorDictDataset):
    """
    Dataset for padded, variable-length sequences (single input/output), auto-truncates batch fields by pad_token_id.

    Args:
        pad_token_id (int): Token ID used for padding (required).
        pad_right (bool): If True, truncates from the right (default). If False, truncates from the left.
        **kwargs: Named tensors (must include input_ids).

    Methods:
        collate_fn(batch): Pads/truncates all fields with 2nd dim matching input_ids.
    """
    def __init__(self, pad_token_id, pad_right=True, **kwargs):
        if 'input_ids' not in kwargs:
            raise Exception("MaskedSequenceDataset requires input_ids")
        super().__init__(**kwargs)
        self.pad_token_id = pad_token_id
        self.pad_right = pad_right
        
    def map(self, f):
        tensors = {}
        for k, v in self._dict.items():
            if isinstance(v, TensorDict):
                tensors[k] = v.map(f)
            elif isinstance(v, torch.Tensor):
                tensors[k] = f(v)
            else:
                raise TypeError(f"Cannot apply map to non-tensor field '{k}' of type {type(v)}")
        return self.__class__(self.pad_token_id, self.pad_right, **tensors)

    def collate_fn(self, batch):
        batch = TensorDict.stack(batch)
        tokens = batch["input_ids"]
        if self.pad_right:
            match = (tokens == self.pad_token_id).all(0)
            if match.any():
                idx = match.byte().argmax(-1)
                for k in batch:
                    if batch[k].ndim == 2 and batch[k].shape[1] == tokens.shape[1]:
                        batch[k] = batch[k][:, :idx]
        else:
            match = (tokens == self.pad_token_id).all(0)
            if match.any():
                idx = match.byte().argmin(-1)
                for k in batch:
                    if batch[k].ndim == 2 and batch[k].shape[1] == tokens.shape[1]:
                        batch[k] = batch[k][:, idx:]
        return batch
        

# class MaskedSequenceDataset(TensorDictDataset):
#     """
#     Dataset for padded, variable-length sequences (single input/output), auto-truncates batch fields by pad_token_id.

#     Args:
#         pad_token_id (int): Token ID used for padding (required).
#         pad_right (bool): If True, truncates from the right (default). If False, truncates from the left.
#         **kwargs: Named tensors (must include input_ids).

#     Methods:
#         collate_fn(batch): Pads/truncates all fields with 2nd dim matching input_ids.
#     """
#     def __init__(self, pad_token_id, pad_right=True, **kwargs):
#         if 'input_ids' not in kwargs:
#             raise Exception("MaskedSequenceDataset requires input_ids")
#         super().__init__(**kwargs)
#         self.pad_token_id = pad_token_id
#         self.pad_right = pad_right

#     def collate_fn(self, batch):
#         batch = TensorDict.stack(batch)
#         tokens = batch["input_ids"]
#         if self.pad_right:
#             idx = (tokens == tokenizer.pad_token_id).all(0).byte().argmax(-1)
#             for k in batch:
#                 if batch[k].ndim == 2 and batch[k].shape[1] == tokens.shape[1]:
#                     batch[k] = batch[k][:, :idx]
#         else:
#             idx = (tokens == self.pad_token_id).all(0).byte().argmin(-1)
#             for k in batch:
#                 if batch[k].ndim == 2 and batch[k].shape[1] == tokens.shape[1]:
#                     batch[k] = batch[k][:, idx:]
#         return batch


class ContrastiveMaskedSequenceDataset(TensorDictDataset):
    """
    Dataset for contrastive learning (positive/negative pairs), truncates each set independently by pad_token_id.

    Args:
        pad_token_id (int): Token ID used for padding (required).
        **kwargs: Named tensors (must include positive_text_tokens, negative_text_tokens).

    Methods:
        collate_fn(batch): Pads/truncates all positive_* and negative_* fields independently.

    Example:
        ds = ContrastiveMaskedSequenceDataset(pad_token_id=0, positive_text_tokens=..., negative_text_tokens=...)
        for batch in ds.to_dataloader(batch_size=8):
            print(batch["positive_text_tokens"].shape, batch["negative_text_tokens"].shape)
    """
    def __init__(self, pad_token_id, **kwargs):
        required = ["positive_text_tokens", "negative_text_tokens"]
        for field in required:
            if field not in kwargs:
                raise Exception(f"{field} is required")
        super().__init__(**kwargs)
        self.pad_token_id = pad_token_id

    def collate_fn(self, batch):
        batch = TensorDict.stack(batch)
        # Truncate positive fields
        pos_tokens = batch["positive_text_tokens"]
        pos_len = (pos_tokens != self.pad_token_id).byte().argmin(-1)
        pos_len[pos_len == 0] = pos_tokens.shape[1]
        new_pos_len = 1 + pos_len.max()
        for k in batch:
            if k.startswith("positive_") and batch[k].ndim == 2:
                batch[k] = batch[k][:, :new_pos_len]
        # Truncate negative fields
        neg_tokens = batch["negative_text_tokens"]
        neg_len = (neg_tokens != self.pad_token_id).byte().argmin(-1)
        neg_len[neg_len == 0] = neg_tokens.shape[1]
        new_neg_len = 1 + neg_len.max()
        for k in batch:
            if k.startswith("negative_") and batch[k].ndim == 2:
                batch[k] = batch[k][:, :new_neg_len]
        return batch
