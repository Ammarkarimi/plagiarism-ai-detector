"""Build a balanced, multi-source human-vs-AI training set from public datasets.

Every source is converted to the same record format::

    {"text": str, "label": 0 | 1, "source": str, "generator": str,
     "domain": str, "attack": str, "group": str}

``label`` is **1 for AI-generated** and 0 for human-written. ``group`` keeps
related texts (for example a human and an AI answer to the same question) in
the same split, so the model can't score well by memorising topics.

Large datasets are sampled by reading random Parquet row groups from the
Hub's Parquet conversion over HTTP, so you never download RAID's full 10+ GB.

Usage::

    plagai prepare-ai-data --out data/ai --sources hc3 mage raid wiki ai_pile \
        --per-source 40000 --ood-source raid
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterator
from pathlib import Path

from plagai.text import count_words, normalize

log = logging.getLogger(__name__)

DATASETS_SERVER = "https://datasets-server.huggingface.co/parquet?dataset={}"

MIN_WORDS = 40
MAX_CHARS = 6000


# ---------------------------------------------------------------------------
# Remote Parquet sampling
# ---------------------------------------------------------------------------

def _parquet_urls(dataset: str, config: str, split: str) -> list[str]:
    with urllib.request.urlopen(DATASETS_SERVER.format(dataset), timeout=60) as r:
        files = json.load(r)["parquet_files"]
    urls = [f["url"] for f in files if f["config"] == config and f["split"] == split]
    if not urls:
        raise ValueError(f"No parquet files for {dataset} [{config}/{split}]")
    return urls


def sample_rows(dataset: str, config: str, split: str, columns: list[str] | None,
                rng: random.Random, rows_per_group: int = 100,
                always_keep: Callable[[dict], bool] | None = None) -> Iterator[dict]:
    """Yield rows from row groups visited in random order across all shards of a split.

    Only a random ``rows_per_group`` slice of each group is used, so a sample
    spans many groups (and generators and domains) instead of a few
    contiguous ones. Rows matching ``always_keep`` (for example the rare human
    rows in RAID) are never dropped. The consumer stops iterating once it has
    enough rows.
    """
    import fsspec
    import pyarrow.parquet as pq

    urls = _parquet_urls(dataset, config, split)
    handles = {}
    groups: list[tuple[str, int]] = []
    for url in urls:
        fh = fsspec.open(url, "rb").open()
        pf = pq.ParquetFile(fh)
        handles[url] = pf
        groups.extend((url, g) for g in range(pf.metadata.num_row_groups))
    rng.shuffle(groups)
    try:
        for url, g in groups:
            rows = handles[url].read_row_group(g, columns=columns).to_pylist()
            rng.shuffle(rows)
            if always_keep:
                yield from (r for r in rows if always_keep(r))
                rows = [r for r in rows if not always_keep(r)]
            yield from rows[:rows_per_group]
    finally:
        for pf in handles.values():
            pf.close()


# ---------------------------------------------------------------------------
# Per-source adapters. Each yields records in the unified format.
# ---------------------------------------------------------------------------

def _rec(text, label, source, generator="human", domain="", attack="none", group=""):
    return {"text": text, "label": int(label), "source": source, "generator": generator,
            "domain": domain, "attack": attack, "group": f"{source}:{group}"}


def load_hc3(rng: random.Random) -> Iterator[dict]:
    """HC3: human vs ChatGPT answers to the same questions."""
    for row in sample_rows("Hello-SimpleAI/HC3", "all", "train", None, rng, rows_per_group=10**9):
        q = str(row.get("id") or row["question"])
        domain = row.get("source", "")
        for t in row["human_answers"] or []:
            yield _rec(t, 0, "hc3", "human", domain, group=q)
        for t in row["chatgpt_answers"] or []:
            yield _rec(t, 1, "hc3", "chatgpt", domain, group=q)


def load_mage(rng: random.Random) -> Iterator[dict]:
    """MAGE (DeepfakeTextDetect): 27 LLMs x 7 domains. NOTE: label 1 = HUMAN upstream.

    ``src`` looks like ``cmv_human`` or ``sci_gen_machine_continuation_opt_125m``.
    """
    for row in sample_rows("yaful/MAGE", "default", "train", ["text", "label", "src"], rng):
        src = row["src"]
        human = int(row["label"]) == 1
        if "_machine_" in src:
            domain, rest = src.split("_machine_", 1)
            gen = rest.split("_", 1)[1] if "_" in rest else rest   # drop "continuation"/"specified"/"topical"
        else:
            domain, gen = src.removesuffix("_human"), "human"
        yield _rec(row["text"], 0 if human else 1, "mage", "human" if human else gen, domain,
                   group=hashlib.md5(row["text"][:200].encode()).hexdigest())


def load_raid(rng: random.Random, include_attacks: bool = True) -> Iterator[dict]:
    """RAID: 11 generators x 8 domains x 11 adversarial attacks x 4 decoding strategies."""
    cols = ["model", "attack", "domain", "generation", "source_id", "adv_source_id", "decoding"]
    for row in sample_rows("liamdugan/raid", "raid", "train", cols, rng,
                           always_keep=lambda r: r["model"] == "human"):
        attack = row["attack"] or "none"
        if not include_attacks and attack != "none":
            continue
        human = row["model"] == "human"
        yield _rec(row["generation"], 0 if human else 1, "raid", row["model"], row["domain"],
                   attack=attack, group=row["source_id"] or row["adv_source_id"])


def load_wiki(rng: random.Random) -> Iterator[dict]:
    """GPT-wiki-intro: real Wikipedia intros paired with GPT (Curie) generated intros."""
    for row in sample_rows("aadityaubhat/GPT-wiki-intro", "default", "train",
                           ["title", "wiki_intro", "generated_intro"], rng):
        yield _rec(row["wiki_intro"], 0, "wiki", "human", "wikipedia", group=row["title"])
        yield _rec(row["generated_intro"], 1, "wiki", "gpt3-curie", "wikipedia", group=row["title"])


def load_ai_pile(rng: random.Random) -> Iterator[dict]:
    """artem9k/ai-text-detection-pile: mixed web/essay text, source in {human, ai}."""
    for row in sample_rows("artem9k/ai-text-detection-pile", "default", "train",
                           ["source", "text", "id"], rng):
        ai = row["source"] == "ai"
        yield _rec(row["text"], int(ai), "ai_pile", "ai" if ai else "human", "mixed", group=str(row["id"]))


def load_csv(path: str) -> Callable[[random.Random], Iterator[dict]]:
    """Local CSV with ``text`` and ``label`` (1 = AI) columns, e.g. Kaggle DAIGT-v2.

    Optional columns: ``source``/``model``/``prompt_name``.
    """
    def _load(rng: random.Random) -> Iterator[dict]:
        import csv
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        rng.shuffle(rows)
        name = Path(path).stem
        for row in rows:
            gen = row.get("model") or row.get("source") or ("ai" if row["label"] == "1" else "human")
            yield _rec(row["text"], int(row["label"]), name, gen, row.get("prompt_name", ""),
                       group=row.get("prompt_name") or hashlib.md5(row["text"][:200].encode()).hexdigest())
    return _load


# Multi-generator sources: cap each generator's share of the AI half.
GENERATOR_CAP = {"mage": 0.15, "raid": 0.2}

LOADERS: dict[str, Callable[..., Iterator[dict]]] = {
    "hc3": load_hc3,
    "mage": load_mage,
    "raid": load_raid,
    "wiki": load_wiki,
    "ai_pile": load_ai_pile,
}


# ---------------------------------------------------------------------------
# Building the dataset
# ---------------------------------------------------------------------------

def _clean(rec: dict) -> dict | None:
    text = normalize(rec["text"] or "")
    if count_words(text) < MIN_WORDS:
        return None
    rec["text"] = text[:MAX_CHARS]
    return rec


def collect(loader: Callable[..., Iterator[dict]], per_class: int, rng: random.Random,
            max_per_generator: float = 1.0) -> list[dict]:
    """Take up to ``per_class`` cleaned records of each label from one source.

    No single AI generator may exceed ``max_per_generator`` of the AI half, so
    one model can't dominate a source.
    """
    out: list[dict] = []
    by_label = Counter()
    by_gen = Counter()
    seen: set[str] = set()
    gen_cap = max(1, int(per_class * max_per_generator))
    for rec in loader(rng):
        rec = _clean(rec)
        if rec is None:
            continue
        lab = rec["label"]
        if by_label[lab] >= per_class:
            if all(by_label[k] >= per_class for k in (0, 1)):
                break
            continue
        if lab == 1 and by_gen[rec["generator"]] >= gen_cap:
            continue
        h = hashlib.sha1(rec["text"].encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        by_label[lab] += 1
        by_gen[rec["generator"]] += lab
        out.append(rec)
    # Rebalance: equal human and AI counts within the source.
    n = min(by_label[0], by_label[1])
    keep = Counter()
    balanced = []
    for rec in out:
        if keep[rec["label"]] < n:
            keep[rec["label"]] += 1
            balanced.append(rec)
    return balanced


def split_of(group: str, val_frac: float, test_frac: float) -> str:
    h = int(hashlib.md5(group.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < test_frac:
        return "test"
    if h < test_frac + val_frac:
        return "val"
    return "train"


def build(sources: list[str], per_source: int, out_dir: str, seed: int = 13,
          val_frac: float = 0.1, test_frac: float = 0.1, ood_sources: list[str] | None = None,
          extra_csv: list[str] | None = None) -> dict:
    rng = random.Random(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ood_sources = set(ood_sources or [])
    loaders = {s: LOADERS[s] for s in sources}
    for path in extra_csv or []:
        loaders[Path(path).stem] = load_csv(path)

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": [], "test_ood": []}
    for name, loader in loaders.items():
        log.info("collecting %s ...", name)
        recs = collect(loader, per_source // 2, rng, GENERATOR_CAP.get(name, 1.0))
        log.info("  %s: %d records", name, len(recs))
        for rec in recs:
            if name in ood_sources:
                splits["test_ood"].append(rec)
            else:
                splits[split_of(rec["group"], val_frac, test_frac)].append(rec)

    stats = {}
    for split, recs in splits.items():
        rng.shuffle(recs)
        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        stats[split] = {
            "n": len(recs),
            "ai": sum(r["label"] for r in recs),
            "by_source": dict(Counter(r["source"] for r in recs)),
        }
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def read_jsonl(path: str | Path, limit: int | None = None) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/ai")
    p.add_argument("--sources", nargs="+", default=list(LOADERS), choices=list(LOADERS))
    p.add_argument("--per-source", type=int, default=40_000, help="records per source (half human, half AI)")
    p.add_argument("--ood-source", nargs="*", default=[], help="sources held out entirely for OOD testing")
    p.add_argument("--extra-csv", nargs="*", default=[], help="local CSVs with text,label columns (1 = AI)")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=13)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    stats = build(a.sources, a.per_source, a.out, a.seed, a.val_frac, a.test_frac, a.ood_source, a.extra_csv)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
