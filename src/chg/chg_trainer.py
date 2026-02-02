import os
from pathlib import Path
from tqdm.auto import trange
import pandas as pd

import torch
from torch import nn
from torch.optim.lr_scheduler import LinearLR

from .chg import CHG
from .misc.tensordict_dataset import MaskedSequenceDataset, ContrastiveMaskedSequenceDataset


class CHGTrainer:
    
    def __init__(self, model, dataset: MaskedSequenceDataset, num_masks=None, mask_init_range=(0, 1),
             batch_size=32, l1_weight=0.1, l1_clamp=4,
             lr=1.0, lr_end=None, lr_reg=1.0, lr_reg_end=None,
             gradient_accum_steps=1, grad_norm=1.0):
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.l1_weight = l1_weight
        self.l1_clamp = l1_clamp
        self.lr = lr
        self.lr_end = lr_end
        self.lr_reg = lr_reg
        self.lr_reg_end = lr_reg_end
        self.gradient_accum_steps = gradient_accum_steps
        self.grad_norm = grad_norm

        self.masked_model = CHG(model)
        self.mask_logits = self.masked_model.create_masks(num_masks=num_masks, mask_init_range=mask_init_range)
        self.mask_logits_pos = nn.Parameter(torch.zeros_like(self.mask_logits))
        self.mask_logits_neg = nn.Parameter(torch.zeros_like(self.mask_logits))

        self.scaler = torch.amp.GradScaler()
        self.iterator = dataset.to_iterator(batch_size=batch_size, shuffle=True)

        self.steps_since_update = 0
        self.update_nll = 0
        self.update_L1 = 0

    def reset_optimizer(self, mask_logits, lr, end_lr=None, total_steps=None):
        self.optimizer = torch.optim.Adam([mask_logits], lr=lr)
        if end_lr is not None:
            self.scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=end_lr / lr,
                total_iters=total_steps
            )
        else:
            self.scheduler = None
        
    def zero_grad(self):
        self.optimizer.zero_grad()
        self.steps_since_update = 0
        self.update_nll = 0
        self.update_L1 = 0
        
    def update(self, mask_logits):
        self.scaler.unscale_(self.optimizer)
        if self.grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(mask_logits, max_norm=self.grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.scheduler is not None:
            self.scheduler.step()
        self.zero_grad()
        
    def get_loss(self, mask_logits, prompts, loss_masks, l1_weight):
        with torch.amp.autocast('cuda'):
            logp = self.masked_model.get_logp(mask_logits.sigmoid(), prompts, loss_masks, agg='mean')
            nll = -logp.mean() / self.gradient_accum_steps
            if l1_weight != 0:
                L1 = l1_weight * mask_logits.clamp(-self.l1_clamp, self.l1_clamp).mean() / self.gradient_accum_steps
            else:
                L1 = 0
            loss = nll - L1
        L1 = L1.mean().item() if isinstance(L1, torch.Tensor) else L1
        return loss, nll.item(), L1
        
    def train_step(self, mask_logits, l1_weight):
        batch = next(self.iterator)
        prompts = batch['input_ids']
        loss_masks = batch['loss_masks']

        loss, nll, L1 = self.get_loss(mask_logits, prompts, loss_masks, l1_weight)
        self.scaler.scale(loss).backward()
        
        self.update_nll += nll
        self.update_L1 += L1
        nll, L1 = self.update_nll, self.update_L1
        self.steps_since_update += 1
        if self.steps_since_update >= self.gradient_accum_steps:
            self.update(mask_logits)
        return nll, L1
    
    def fit_mask(self, regularization, num_updates, show_pbar=True):
        if regularization == 'none':
            mask_logits = self.mask_logits
            l1_weight = 0
            lr, end_lr = self.lr, self.lr_end
        elif regularization == 'positive':
            mask_logits = self.mask_logits_pos
            l1_weight = self.l1_weight
            lr, end_lr = self.lr_reg, self.lr_reg_end
        elif regularization == 'negative':
            mask_logits = self.mask_logits_neg
            l1_weight = -self.l1_weight
            lr, end_lr = self.lr_reg, self.lr_reg_end
        else:
            raise ValueError(f"Unknown mask type: {regularization}")

        self.reset_optimizer(mask_logits=mask_logits, lr=lr, end_lr=end_lr, total_steps=num_updates)
    
        self.zero_grad()
        updates = 0
        for step in trange(num_updates*self.gradient_accum_steps, disable=not show_pbar):
            nll, L1 = self.train_step(mask_logits, l1_weight)
            if step == 0 or self.steps_since_update == 0:
                updates += 1
                mask = mask_logits.sigmoid().detach().cpu()
                values = {
                    'regularization': regularization,
                    'num_update': updates,
                    'nll': nll,
                    'L1': L1
                }
                yield mask, values
                
    def fit(self, num_updates, num_reg_updates, masks_savepath=None, verbose=True):
        if masks_savepath is not None:
            if isinstance(masks_savepath, str):
                assert masks_savepath.endswith(".pt"), "masks_savepath must end with .pt"
                os.makedirs(os.path.dirname(masks_savepath), exist_ok=True)
            elif isinstance(masks_savepath, Path):
                assert masks_savepath.suffix == ".pt", "masks_savepath must end with .pt"
                masks_savepath.parent.mkdir(parents=True, exist_ok=True)
            else:
                raise TypeError("masks_savepath must be a string or a Path")
        
        all_metrics = []
        all_masks = {}
        for reg, steps in zip(["none", "positive", "negative"], [num_updates, num_reg_updates, num_reg_updates]):
            if verbose:
                print(f"Fitting masks with regularization: {reg}")
            masks = []
            for mask, metrics in self.fit_mask(reg, steps, show_pbar=verbose):
                all_metrics.append(metrics)
                masks.append(mask)
                yield mask, metrics
            all_masks[reg] = torch.stack(masks, dim=0)
                
            del self.optimizer
            del self.scheduler
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
                
            if reg == 'none':
                with torch.no_grad():
                    self.mask_logits_pos.copy_(self.mask_logits)
                    self.mask_logits_neg.copy_(self.mask_logits)
            if masks_savepath is not None:
                torch.save(all_masks, masks_savepath)


class ContrastiveCHGTrainer:
    """
    Trainer for Contrastive CHG (CCHG).

    Fits a mask that retains performance on positive examples while forgetting negative examples.
    Used to isolate sub-circuits for specific sub-tasks (e.g., ICL vs instruction-following).

    Loss: -logp_pos + (logp_neg - logp_pos.detach()).clamp(min=-max_diff) + L1
    """

    def __init__(self, model, dataset: ContrastiveMaskedSequenceDataset,
                 num_masks: int | None = None, mask_init_range: tuple[float, float] = (0, 1),
                 batch_size: int = 64, l1_weight: float = 0.1, l1_clamp: float = 4, max_diff: float = 5,
                 lr: float = 0.1, lr_end: float | None = 0.01,
                 gradient_accum_steps: int = 1, grad_norm: float = 1.0):
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.l1_weight = l1_weight
        self.l1_clamp = l1_clamp
        self.max_diff = max_diff
        self.lr = lr
        self.lr_end = lr_end
        self.gradient_accum_steps = gradient_accum_steps
        self.grad_norm = grad_norm

        self.masked_model = CHG(model)
        self.mask_logits = self.masked_model.create_masks(num_masks=num_masks, mask_init_range=mask_init_range)

        self.scaler = torch.amp.GradScaler()
        self.iterator = dataset.to_iterator(batch_size=batch_size, shuffle=True)

        self.steps_since_update = 0
        self.update_metrics = {}

    def reset_optimizer(self, lr: float, end_lr: float | None, total_steps: int):
        self.optimizer = torch.optim.Adam([self.mask_logits], lr=lr)
        if end_lr is not None:
            self.scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=end_lr / lr,
                total_iters=total_steps
            )
        else:
            self.scheduler = None

    def zero_grad(self):
        self.optimizer.zero_grad()
        self.steps_since_update = 0
        self.update_metrics = {}

    def update(self):
        self.scaler.unscale_(self.optimizer)
        if self.grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.mask_logits, max_norm=self.grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.scheduler is not None:
            self.scheduler.step()
        self.zero_grad()

    def get_loss(self, pos_tokens, pos_masks, neg_tokens, neg_masks):
        with torch.amp.autocast('cuda'):
            mask = self.mask_logits.sigmoid()
            logp_pos = self.masked_model.get_logp(mask, pos_tokens, pos_masks, agg='mean')
            logp_neg = self.masked_model.get_logp(mask, neg_tokens, neg_masks, agg='mean')

            diff = logp_neg - logp_pos.detach()
            diff_loss = diff.clamp(min=-self.max_diff).mean()

            nll_pos = -logp_pos.mean() / self.gradient_accum_steps
            diff_loss = diff_loss / self.gradient_accum_steps
            L1 = self.l1_weight * self.mask_logits.clamp(-self.l1_clamp, self.l1_clamp).mean() / self.gradient_accum_steps

            loss = nll_pos + diff_loss - L1  # negative L1 encourages sparsity

        return loss, {
            'logp_pos': logp_pos.mean().item(),
            'logp_neg': logp_neg.mean().item(),
            'diff': diff.mean().item(),
            'L1': L1.item()
        }

    def train_step(self):
        batch = next(self.iterator)
        pos_tokens = batch['positive_text_tokens']
        pos_masks = batch['positive_loss_masks']
        neg_tokens = batch['negative_text_tokens']
        neg_masks = batch['negative_loss_masks']

        loss, metrics = self.get_loss(pos_tokens, pos_masks, neg_tokens, neg_masks)
        self.scaler.scale(loss).backward()

        self.steps_since_update += 1
        for k, v in metrics.items():
            if k not in self.update_metrics:
                self.update_metrics[k] = 0
            self.update_metrics[k] += v / self.gradient_accum_steps

        if self.steps_since_update >= self.gradient_accum_steps:
            self.update()
        return self.update_metrics.copy()

    def fit(self, num_updates: int, mask_savepath: str | Path | None = None, show_pbar: bool = True):
        """
        Fit the contrastive mask.

        Args:
            num_updates: Number of optimizer updates.
            mask_savepath: Optional path to save the fitted mask (.pt file).
            show_pbar: Whether to show progress bar.

        Yields:
            Tuple of (mask, metrics) after each update.
        """
        if mask_savepath is not None:
            mask_savepath = Path(mask_savepath)
            mask_savepath.parent.mkdir(parents=True, exist_ok=True)

        self.reset_optimizer(lr=self.lr, end_lr=self.lr_end, total_steps=num_updates)
        self.zero_grad()

        updates = 0
        for step in trange(num_updates * self.gradient_accum_steps, disable=not show_pbar):
            metrics = self.train_step()
            if step == 0 or self.steps_since_update == 0:
                updates += 1
                mask = self.mask_logits.sigmoid().detach().cpu()
                metrics['num_update'] = updates
                metrics['mask_min'] = mask.min().item()
                metrics['mask_mean'] = mask.mean().item()
                metrics['mask_max'] = mask.max().item()
                yield mask, metrics

        if mask_savepath is not None:
            torch.save(self.mask_logits.sigmoid().cpu(), mask_savepath)
