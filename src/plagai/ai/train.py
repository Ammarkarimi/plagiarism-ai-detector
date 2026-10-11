"""Fine-tune a transformer encoder (default DeBERTa-v3) as a human-vs-AI classifier.

Recipe (see docs/RESEARCH.md):
  * multi-source balanced data from ``plagai prepare-ai-data``
  * random-crop + adversarial augmentation applied equally to both classes
  * model selection on validation AUROC, then reporting of TPR@1%FPR on test and OOD splits

Example (single 24 GB GPU, about 3-5 h for 200k texts)::

    plagai train-ai --data data/ai --model microsoft/deberta-v3-base \
        --out artifacts/ai-detector --epochs 2 --batch-size 16 --max-length 512 --bf16

Quick CPU smoke test::

    plagai train-ai --data data/ai_small --model google/electra-small-discriminator \
        --out artifacts/ai-detector-smoke --max-train 1500 --max-eval 400 --max-length 256 --epochs 1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          EarlyStoppingCallback, Trainer, TrainingArguments, set_seed)

from plagai.ai.augment import augment, random_crop
from plagai.ai.data import read_jsonl
from plagai.ai.metrics import breakdown, summarize

log = logging.getLogger(__name__)


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_length: int, train: bool, augment_p: float = 0.0):
        self.rows = rows
        self.tok = tokenizer
        self.max_length = max_length
        self.train = train
        self.augment_p = augment_p
        # About 1.3 tokens per English word; crop so most windows fit without truncation.
        self.crop_words = int(max_length / 1.35)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        text = r["text"]
        if self.train:
            # DataLoader re-seeds the global `random` per worker, so draw from it instead of
            # a shared instance (which every worker would copy with identical state).
            rng = random.Random(random.getrandbits(64))
            text = random_crop(text, rng, self.crop_words)
            text = augment(text, rng, self.augment_p)
        enc = self.tok(text, truncation=True, max_length=self.max_length)
        enc["labels"] = int(r["label"])
        return enc


def resolve_precision(bf16: bool, fp16: bool) -> tuple[bool, bool]:
    """Return (bf16, fp16) flags the current hardware can actually run."""
    if not (bf16 or fp16):
        return False, False
    if not torch.cuda.is_available():
        log.warning("No CUDA GPU visible to PyTorch (build %s); training in fp32 on CPU. "
                    "If you have an NVIDIA GPU, install the CUDA build of torch.", torch.__version__)
        return False, False
    if bf16 and torch.cuda.is_bf16_supported():
        return True, False
    if bf16:
        log.warning("GPU %s has no bf16 support; using fp16 instead.", torch.cuda.get_device_name(0))
    return False, True


def check_disk(out: Path, model, save_optimizer: bool) -> None:
    """Fail fast if the checkpoints (best + latest + final model) can't fit on disk."""
    model_gb = sum(p.numel() for p in model.parameters()) * 4 / 1e9
    need_gb = model_gb * (3 if save_optimizer else 1) * 2 + model_gb + 0.5
    out.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(out).free / 1e9
    log.info("disk: %.1f GB free, about %.1f GB needed for checkpoints", free_gb, need_gb)
    if free_gb < need_gb:
        raise SystemExit(f"Not enough disk space in {out}: {free_gb:.1f} GB free, need about {need_gb:.1f} GB. "
                         "Free some space or pass --out on another drive.")


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    logits = logits[0] if isinstance(logits, tuple) else logits
    scores = logits[:, 1] - logits[:, 0]
    m = summarize(labels, scores, threshold=0.0)
    return {"auroc": m.get("auroc", 0.0), "tpr_at_1fpr": m.get("tpr@1%fpr", 0.0),
            "accuracy": m["accuracy"], "f1": m["f1"]}


def evaluate_split(model_dir: str, rows: list[dict], max_length: int, batch_size: int) -> dict:
    """Full-length, chunked evaluation with the inference wrapper (same code path as the API)."""
    from plagai.ai.classifier import TransformerClassifier

    clf = TransformerClassifier(model_dir, max_length=max_length, batch_size=batch_size)
    probs = clf.predict_proba([r["text"] for r in rows])
    return {
        "overall": summarize([r["label"] for r in rows], probs, 0.5),
        "by_source": breakdown(rows, probs, "source"),
        "by_generator": breakdown(rows, probs, "generator"),
        "by_attack": breakdown(rows, probs, "attack"),
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/ai")
    p.add_argument("--model", default="microsoft/deberta-v3-base")
    p.add_argument("--out", default="artifacts/ai-detector")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--warmup-ratio", type=float, default=0.06)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--augment-p", type=float, default=0.3)
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--max-eval", type=int, default=None)
    p.add_argument("--eval-steps", type=int, default=1000)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--num-workers", type=int, default=0 if sys.platform == "win32" else 2,
                   help="DataLoader workers (default 0 on Windows, where worker spawn is slow)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report-to", default="none", help="none | wandb | tensorboard")
    p.add_argument("--skip-final-eval", action="store_true")
    p.add_argument("--save-optimizer", action="store_true",
                   help="also checkpoint optimizer state (needed only to resume; about 3x the disk)")
    p.add_argument("--keep-checkpoints", action="store_true",
                   help="keep intermediate checkpoints after training (the final model is saved separately)")
    p.add_argument("--eval-only", action="store_true", help="evaluate the model already saved in --out")
    a = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    set_seed(a.seed)
    a.bf16, a.fp16 = resolve_precision(a.bf16, a.fp16)
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    log.info("device=%s bf16=%s fp16=%s", device, a.bf16, a.fp16)
    data = Path(a.data)
    if not a.eval_only:
        train(a, data)
    if not a.skip_final_eval:
        final_eval(a, data)


def train(a: argparse.Namespace, data: Path) -> None:
    train_rows = read_jsonl(data / "train.jsonl", a.max_train)
    val_rows = read_jsonl(data / "val.jsonl", a.max_eval)
    log.info("train=%d val=%d", len(train_rows), len(val_rows))

    tok = AutoTokenizer.from_pretrained(a.model)
    # Keep fp32 master weights: transformers 5 otherwise loads the checkpoint's dtype (fp16 for
    # deberta-v3-base), and AdamW on fp16 weights diverges. --bf16/--fp16 add autocast on top.
    model = AutoModelForSequenceClassification.from_pretrained(
        a.model, num_labels=2, id2label={0: "human", 1: "ai"}, label2id={"human": 0, "ai": 1},
        dtype=torch.float32)
    if a.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    train_ds = TextDataset(train_rows, tok, a.max_length, train=True, augment_p=a.augment_p)
    val_ds = TextDataset(val_rows, tok, a.max_length, train=False)

    out = Path(a.out)
    check_disk(out, model, a.save_optimizer)
    steps_per_epoch = max(1, len(train_ds) // (a.batch_size * a.grad_accum))
    eval_steps = min(a.eval_steps, steps_per_epoch)
    warmup_steps = int(a.warmup_ratio * steps_per_epoch * a.epochs)  # warmup_ratio was removed in transformers 5
    args = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=a.epochs,
        learning_rate=a.lr,
        per_device_train_batch_size=a.batch_size,
        per_device_eval_batch_size=a.batch_size * 2,
        gradient_accumulation_steps=a.grad_accum,
        warmup_steps=warmup_steps,
        weight_decay=a.weight_decay,
        lr_scheduler_type="linear",
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=1,             # the best checkpoint is always kept in addition
        save_only_model=not a.save_optimizer,
        load_best_model_at_end=True,
        metric_for_best_model="auroc",
        greater_is_better=True,
        logging_steps=max(1, eval_steps // 10),
        bf16=a.bf16,
        fp16=a.fp16,
        dataloader_num_workers=a.num_workers,
        report_to=[] if a.report_to == "none" else [a.report_to],
        seed=a.seed,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tok,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
    )
    trainer.train()
    trainer.save_model(str(out))
    tok.save_pretrained(str(out))
    meta = {"base_model": a.model, "max_length": a.max_length, "train_size": len(train_rows),
            "augment_p": a.augment_p, "epochs": a.epochs, "lr": a.lr,
            "val_metrics": trainer.evaluate()}
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2))
    if not a.keep_checkpoints:
        shutil.rmtree(out / "checkpoints", ignore_errors=True)
    log.info("saved model to %s", out)


def final_eval(a: argparse.Namespace, data: Path) -> None:
    out = Path(a.out)
    report = {}
    for split in ("test", "test_ood"):
        path = data / f"{split}.jsonl"
        if path.exists() and path.stat().st_size:
            rows = read_jsonl(path, a.max_eval)
            report[split] = evaluate_split(str(out), rows, a.max_length, a.batch_size * 2)
            log.info("%s: %s", split, json.dumps(report[split]["overall"]))
    (out / "classifier_metrics.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    np.set_printoptions(precision=4)
    main()
