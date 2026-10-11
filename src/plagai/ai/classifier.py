"""Inference wrapper for the fine-tuned sequence classifier, with sliding-window chunking."""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from plagai.config import resolve_device


class TransformerClassifier:
    """Scores texts of any length by averaging logits over overlapping token windows."""

    def __init__(self, model_dir: str, device: str = "auto", max_length: int = 512,
                 batch_size: int = 16, max_windows: int = 16):
        self.device = resolve_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir, dtype=torch.float32)
        self.model.eval().to(self.device)
        if self.device == "cuda":
            # DeBERTa can overflow in fp16; prefer bf16 where the GPU supports it.
            self.model.to(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        self.max_length = min(max_length, getattr(self.tokenizer, "model_max_length", max_length) or max_length)
        self.batch_size = batch_size
        self.max_windows = max_windows
        label2id = {k.lower(): v for k, v in (self.model.config.label2id or {}).items()}
        self.ai_index = label2id.get("ai", 1)
        self.prefix, self.suffix = self._special_tokens()
        self.pad_id = self.tokenizer.pad_token_id or 0

    def _special_tokens(self) -> tuple[list[int], list[int]]:
        """Find the special tokens the tokenizer wraps around a single sequence (e.g. [CLS] ... [SEP]).

        Works across tokenizer families and transformers versions, unlike
        ``build_inputs_with_special_tokens``, which transformers 5 removed.
        """
        plain = self.tokenizer("hello", add_special_tokens=False)["input_ids"]
        full = self.tokenizer("hello")["input_ids"]
        for i in range(len(full) - len(plain) + 1):
            if full[i:i + len(plain)] == plain:
                return full[:i], full[i + len(plain):]
        return [], []

    def _windows(self, text: str) -> list[list[int]]:
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        size = self.max_length - len(self.prefix) - len(self.suffix)
        if len(ids) <= size:
            return [ids]
        stride = int(size * 0.75)
        starts = list(range(0, len(ids) - size + 1, stride))
        if starts[-1] + size < len(ids):
            starts.append(len(ids) - size)
        if len(starts) > self.max_windows:  # spread windows evenly over very long documents
            starts = [starts[i] for i in np.linspace(0, len(starts) - 1, self.max_windows).astype(int)]
        return [ids[s:s + size] for s in starts]

    @torch.inference_mode()
    def _logits(self, windows: list[list[int]]) -> np.ndarray:
        out = []
        for i in range(0, len(windows), self.batch_size):
            seqs = [self.prefix + w + self.suffix for w in windows[i:i + self.batch_size]]
            width = max(map(len, seqs))
            ids = torch.full((len(seqs), width), self.pad_id, dtype=torch.long)
            mask = torch.zeros((len(seqs), width), dtype=torch.long)
            for r, seq in enumerate(seqs):  # right padding
                ids[r, :len(seq)] = torch.tensor(seq)
                mask[r, :len(seq)] = 1
            logits = self.model(input_ids=ids.to(self.device), attention_mask=mask.to(self.device)).logits.float()
            # Log-odds of the AI class, valid for 2-class softmax heads.
            out.append((logits[:, self.ai_index] - logits[:, 1 - self.ai_index]).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    def predict_logit(self, texts: list[str]) -> np.ndarray:
        """AI log-odds per text (length-weighted mean over windows)."""
        all_windows, owners, weights = [], [], []
        for j, t in enumerate(texts):
            for w in self._windows(t):
                all_windows.append(w)
                owners.append(j)
                weights.append(max(len(w), 1))
        logits = self._logits(all_windows)
        owners_a = np.array(owners)
        weights_a = np.array(weights, dtype=float)
        res = np.zeros(len(texts), dtype=np.float32)
        for j in range(len(texts)):
            m = owners_a == j
            res[j] = np.average(logits[m], weights=weights_a[m]) if m.any() else 0.0
        return res

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.predict_logit(texts)))
