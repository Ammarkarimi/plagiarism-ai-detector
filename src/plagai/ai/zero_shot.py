"""Binoculars zero-shot detector (Hans et al., ICML 2024, arXiv:2401.12070).

    score = log-perplexity_performer(x) / cross-perplexity(observer || performer)(x)

A low score means the text is "surprisingly unsurprising" for its content,
which is the signature of LLM text. Both models must share a tokenizer. The
paper used falcon-7b (observer) and falcon-7b-instruct (performer); any base
and instruct pair works, for example Qwen2.5-1.5B and Qwen2.5-1.5B-Instruct.
No training is needed. We feed the raw score, plus perplexity statistics,
into the ensemble.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from plagai.config import resolve_device

ZS_FEATURES = ["bino_score", "log_ppl", "log_xppl"]


class Binoculars:
    def __init__(self, observer: str, performer: str, device: str = "auto", max_tokens: int = 512,
                 batch_size: int = 8):
        self.device = resolve_device(device)
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(observer)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "right"
        self.observer = AutoModelForCausalLM.from_pretrained(observer, dtype=dtype).to(self.device).eval()
        self.performer = AutoModelForCausalLM.from_pretrained(performer, dtype=dtype).to(self.device).eval()
        self.max_tokens = max_tokens
        self.batch_size = batch_size

    @torch.inference_mode()
    def _batch(self, texts: list[str]) -> np.ndarray:
        enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True,
                       max_length=self.max_tokens).to(self.device)
        obs = self.observer(**enc).logits[:, :-1].float()
        perf = self.performer(**enc).logits[:, :-1].float()
        target = enc["input_ids"][:, 1:]
        mask = enc["attention_mask"][:, 1:].float()
        n = mask.sum(1).clamp(min=1)

        # Log-perplexity of the text under the performer.
        nll = F.cross_entropy(perf.transpose(1, 2), target, reduction="none")
        log_ppl = (nll * mask).sum(1) / n
        # Cross-perplexity: expected performer loss under the observer's next-token distribution.
        x_ent = -(F.softmax(obs, -1) * F.log_softmax(perf, -1)).sum(-1)
        log_xppl = (x_ent * mask).sum(1) / n
        score = log_ppl / log_xppl.clamp(min=1e-6)
        return torch.stack([score, log_ppl, log_xppl], 1).cpu().numpy()

    def features(self, texts: list[str]) -> np.ndarray:
        out = [self._batch(texts[i:i + self.batch_size]) for i in range(0, len(texts), self.batch_size)]
        return np.concatenate(out) if out else np.zeros((0, len(ZS_FEATURES)), dtype=np.float32)

    def score(self, texts: list[str]) -> np.ndarray:
        return self.features(texts)[:, 0]
