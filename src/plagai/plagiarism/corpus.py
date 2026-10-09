"""Per-tenant corpus namespaces, plus a CLI to bulk-index reference documents.

Namespaces:
  * ``public``: the shared reference corpus you curate (Wikipedia, arXiv, open textbooks and similar)
  * ``acct-<id>``: each customer's private corpus (their past submissions), never shared

Usage::

    plagai index-corpus --input refs/ --namespace public          # *.txt files
    plagai index-corpus --input wiki.jsonl --namespace public     # {"text","title","url"} per line
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import threading
from pathlib import Path

from plagai.plagiarism.embeddings import Encoder, build_encoder
from plagai.plagiarism.index import CorpusIndex

log = logging.getLogger(__name__)
_SAFE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class CorpusManager:
    def __init__(self, root: str | Path, encoder: Encoder):
        self.root = Path(root)
        self.encoder = encoder
        self._indexes: dict[str, CorpusIndex] = {}
        self._lock = threading.Lock()

    def get(self, namespace: str) -> CorpusIndex:
        if not _SAFE.match(namespace):
            raise ValueError("invalid namespace")
        with self._lock:
            if namespace not in self._indexes:
                self._indexes[namespace] = CorpusIndex.load_or_create(self.root / namespace, self.encoder)
            return self._indexes[namespace]

    def save(self, namespace: str) -> None:
        self.get(namespace).save(self.root / namespace)


def _iter_docs(path: Path):
    if path.is_dir():
        for f in sorted(path.rglob("*.txt")):
            yield {"text": f.read_text(encoding="utf-8", errors="ignore"), "title": f.stem, "url": str(f)}
    elif path.suffix == ".jsonl":
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                yield json.loads(line)
    else:
        yield {"text": path.read_text(encoding="utf-8", errors="ignore"), "title": path.stem, "url": str(path)}


def main(argv: list[str] | None = None) -> None:
    from plagai.config import get_settings

    s = get_settings()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--namespace", default="public")
    p.add_argument("--corpus-dir", default=s.corpus_dir)
    p.add_argument("--encoder", default=s.embedding_model)
    p.add_argument("--save-every", type=int, default=5000)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    mgr = CorpusManager(a.corpus_dir, build_encoder(a.encoder))
    idx = mgr.get(a.namespace)
    n = 0
    for doc in _iter_docs(Path(a.input)):
        if not doc.get("text"):
            continue
        idx.add(doc["text"], doc.get("title", ""), doc.get("url", ""), doc.get("id"))
        n += 1
        if n % a.save_every == 0:
            mgr.save(a.namespace)
            log.info("indexed %d documents", n)
    mgr.save(a.namespace)
    print(json.dumps({"namespace": a.namespace, "indexed": n, "total_docs": len(idx)}))


if __name__ == "__main__":
    main()
