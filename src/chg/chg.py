import contextlib
from functools import partial
import einops
import torch
from torch import nn
from .misc import torch_utils as tu


class CHG:
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.mask_hooks = []
        
    @property
    def device(self):
        return next(self.model.parameters()).device
    
    @property
    def layers(self):
        return self.model.model.layers
    
    @property
    def num_layers(self):
        return self.model.config.num_hidden_layers
    
    @property
    def num_heads(self):
        return self.model.config.num_attention_heads
    
    @property
    def mask_shape(self):
        return (self.num_layers, self.num_heads)
    
    def create_masks(self, num_masks=None, mask_init_range=(0.0, 1.0), parameters=True, logits=True):
        if num_masks is None:
            masks = torch.rand(self.num_layers, self.num_heads).to(self.device)
        else:
            masks = torch.rand(num_masks, self.num_layers, self.num_heads).to(self.device)
        masks = mask_init_range[0] + (mask_init_range[1] - mask_init_range[0]) * masks
        if logits:
            masks = (masks / (1 - masks)).log()
        if parameters:
            masks = nn.Parameter(masks)
        return masks
    
    def remove_hooks(self):
        for hook in self.mask_hooks:
            hook.remove()
        self.mask_hooks.clear()

    @contextlib.contextmanager
    def register_mask_hooks(self, masks):
        """Registers pre-W_O hooks and ensures they are removed after execution."""
        def pre_hook(module, input, layer_idx=None):
            x = input[0]  # [B, L, D]
            bsz, seq_len, dim_h = x.shape
            dim_head = dim_h // self.num_heads
            x = x.view(bsz, seq_len, self.num_heads, dim_head)
            mask = masks[:, layer_idx].view(-1, 1, self.num_heads, 1).to(x.device)
            x = x * mask
            x = x.view(bsz, seq_len, dim_h)
            return (x,)

        for i, layer in enumerate(self.layers):
            hook = layer.self_attn.o_proj.register_forward_pre_hook(partial(pre_hook, layer_idx=i))
            self.mask_hooks.append(hook)

        try:
            yield
        finally:
            self.remove_hooks()

    def forward(self, inputs, masks=None):
        """
        inputs: tensor with shape [batch_size, seq_length]
        masks: binary tensor with shape [num_layers, num_heads] or [num_masks, num_layers, num_heads]
        return: tensor with shape [batch_size, seq_length] or [batch_size, num_masks, seq_length]
        """
        if masks is None:
            return self.model(inputs).logits

        single_mask = False
        if masks.ndim == 2:
            masks = masks.unsqueeze(0)
            single_mask = True

        batch_size = inputs.size(0)
        num_masks = masks.size(0)

        inputs = einops.repeat(inputs, 'b s -> (b k) s', k=num_masks)
        masks = einops.repeat(masks, 'k l h -> (b k) l h', b=batch_size)

        with self.register_mask_hooks(masks):
            output = self.model(inputs).logits
            output = output.view(batch_size, num_masks, *output.shape[1:])
            return output[:, 0] if single_mask else output

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    def get_logp(self, mask, text_tokens, loss_masks, agg='sum', return_accuracy=False):
        """
        Computes the total log-probability of target tokens under a masked model,
        supporting both a single mask ([m, n]) or multiple masks ([b, m, n]).

        Args:
            mask (Tensor): Attention mask of shape [m, n] or [b, m, n].
            text_tokens (Tensor): Input token tensor of shape [b, seq_len].
            loss_masks (Tensor): Binary mask of shape [b, seq_len] indicating which tokens to include in loss.
            agg (str): Aggregation method: 'none', 'sum', or 'mean'.
            return_accuracy (bool): If True, also return accuracy over masked tokens.

        Returns:
            Tensor: Log-probabilities of shape [b] or [b, m].
            Tensor (optional): Accuracy values of shape [b] or [b, m] if return_accuracy is True.
        """
        assert agg in ['none', 'sum', 'mean'], f"Unsupported aggregation method: {agg}"
        
        single_mask = False
        if mask.dim() == 2:
            single_mask = True
            mask = mask.unsqueeze(0)
        
        logits = self(text_tokens[:, :-1], mask)
        targets = einops.repeat(text_tokens[:, 1:], 'b l -> b m l', m=len(mask))
        loss_masks = einops.repeat(loss_masks[:, 1:], 'b l -> b m l', m=len(mask))

        if return_accuracy:
            pred = logits.argmax(-1)
            correct = (pred == targets) * loss_masks
            accuracy = correct.sum(-1) / loss_masks.sum(-1)

        nll = tu.cross_entropy(logits, targets, reduction='none')
        logp = -(nll * loss_masks)

        if agg != 'none':
            logp = logp.sum(dim=-1)
        if agg == 'mean':
            logp = logp / loss_masks.sum(dim=-1)

        if single_mask:
            logp = logp.squeeze(1)
            if return_accuracy:
                return logp, accuracy.squeeze(1)
        if return_accuracy:
            return logp, accuracy
        return logp
