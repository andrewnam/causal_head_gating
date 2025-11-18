import torch
from torch.nn import functional as F


class PromptTokenizer:
    def __init__(self, tokenizer, device='cpu', marker=' <|TARGET|> '):
        if marker[0] != ' ':
            marker = ' ' + marker
        if marker[-1] != ' ':
            marker = marker + ' '
        if marker not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({'additional_special_tokens': [marker]})
        self.tokenizer = tokenizer
        self.device = device
        self.marker = marker
        self.marker_id = tokenizer(marker, add_special_tokens=False)['input_ids'][0]

    def tokenize(self, prompt: str, target: str):
        prompt = prompt[:-1] if prompt.endswith(' ') else prompt
        target = ' ' + target if not target.startswith(' ') else target
        full_text = prompt + self.marker + target
        input_ids = self.tokenizer(full_text, add_special_tokens=False)['input_ids']
        idx = input_ids.index(self.marker_id)
        new_ids = input_ids[:idx] + input_ids[idx + 1:]
        mask = [False] * idx + [True] * (len(new_ids) - idx)
        return (
            torch.tensor(new_ids, dtype=torch.long, device=self.device),
            torch.tensor(mask, dtype=torch.bool, device=self.device),
        )

    def tokenize_batch(self, prompts: list[str], targets: list[str]):
        prompts = [p[:-1] if p.endswith(' ') else p for p in prompts]
        targets = [' ' + t if not t.startswith(' ') else t for t in targets]
        full_texts = [p + self.marker + t for p, t in zip(prompts, targets)]
        tokenized = self.tokenizer(full_texts, padding=False, truncation=True, add_special_tokens=False)
        input_ids_list = tokenized['input_ids']

        input_ids_trimmed, mask_trimmed = [], []
        for ids in input_ids_list:
            idx = ids.index(self.marker_id)
            new_ids = ids[:idx] + ids[idx + 1:]
            mask = [False] * idx + [True] * (len(new_ids) - idx)
            input_ids_trimmed.append(torch.tensor(new_ids, dtype=torch.long))
            mask_trimmed.append(torch.tensor(mask, dtype=torch.bool))

        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids_trimmed, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        mask_padded = torch.nn.utils.rnn.pad_sequence(
            mask_trimmed, batch_first=True, padding_value=False
        )

        input_ids_padded = F.pad(input_ids_padded, (0, 1), value=self.tokenizer.pad_token_id)
        mask_padded = F.pad(mask_padded, (0, 1), value=False)
        return input_ids_padded.to(self.device), mask_padded.to(self.device)
