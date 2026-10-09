"""Fine-tune the sentence encoder for paraphrase-robust plagiarism matching.

Data: PAWS (adversarial word-swap pairs) and MRPC (news paraphrases), both
with binary paraphrase labels. Loss: OnlineContrastiveLoss, which pulls
paraphrases together and pushes hard non-paraphrases apart. That is what we
need, because we threshold cosine similarity at inference time.

The script also reports the cosine threshold that maximises F1 on the dev
set. Copy that value into ``PLAGAI_PLAG_PARAPHRASE_THRESHOLD``.

Usage::

    plagai train-paraphrase --base sentence-transformers/all-MiniLM-L6-v2 \
        --out artifacts/paraphrase-encoder --epochs 1
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def load_pairs(max_train: int | None, max_eval: int | None):
    from datasets import Value, concatenate_datasets, load_dataset

    paws = load_dataset("google-research-datasets/paws", "labeled_final")
    mrpc = load_dataset("nyu-mll/glue", "mrpc")

    def norm(ds):
        # Both use label 1 = paraphrase; drop the ClassLabel types so they concatenate.
        return ds.select_columns(["sentence1", "sentence2", "label"]).cast_column("label", Value("int64"))

    train = concatenate_datasets([norm(paws["train"]), norm(mrpc["train"])]).shuffle(seed=0)
    dev = concatenate_datasets([norm(paws["validation"]), norm(mrpc["validation"])]).shuffle(seed=0)
    if max_train:
        train = train.select(range(min(max_train, len(train))))
    if max_eval:
        dev = dev.select(range(min(max_eval, len(dev))))
    return train, dev


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--out", default="artifacts/paraphrase-encoder")
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--max-eval", type=int, default=None)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments)
    from sentence_transformers.evaluation import BinaryClassificationEvaluator
    from sentence_transformers.losses import OnlineContrastiveLoss

    train, dev = load_pairs(a.max_train, a.max_eval)
    log.info("train pairs=%d dev pairs=%d", len(train), len(dev))
    model = SentenceTransformer(a.base)
    evaluator = BinaryClassificationEvaluator(dev["sentence1"], dev["sentence2"], dev["label"], name="dev")
    before = evaluator(model)

    steps = int(len(train) / a.batch_size * a.epochs)
    args = SentenceTransformerTrainingArguments(
        output_dir=str(Path(a.out) / "checkpoints"), num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch_size, learning_rate=a.lr, warmup_steps=int(0.1 * steps),
        eval_strategy="no", save_strategy="no", logging_steps=50, report_to=[])
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=train,
                                         loss=OnlineContrastiveLoss(model), evaluator=evaluator)
    trainer.train()
    after = evaluator(model)
    model.save(a.out)
    key = next((k for k in after if k.endswith("cosine_f1_threshold")), None)
    report = {"before": before, "after": after, "recommended_paraphrase_threshold": after.get(key)}
    (Path(a.out) / "paraphrase_metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
