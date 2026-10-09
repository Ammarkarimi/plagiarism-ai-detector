"""Web source retrieval: search distinctive sentences, fetch the pages and index them on the fly.

Ships with a Brave Search API provider (set ``PLAGAI_BRAVE_API_KEY``).
Implement ``SearchProvider`` to plug in Bing, Google Programmable Search,
SerpAPI or your own crawl.
"""

from __future__ import annotations

import html
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

import httpx

from plagai.plagiarism.embeddings import Encoder
from plagai.plagiarism.index import CorpusIndex
from plagai.text import split_sentences, words

log = logging.getLogger(__name__)

_SCRIPT = re.compile(r"<(script|style|noscript|svg|nav|footer|header)[^>]*>.*?</\1>", re.S | re.I)
_TAG = re.compile(r"<[^>]+>")
_BLOCK = re.compile(r"</?(p|div|br|li|h[1-6]|tr|section|article)[^>]*>", re.I)


class SearchProvider(Protocol):
    def search(self, query: str, count: int = 5) -> list[dict]:
        """Return a list of {"url", "title"} results."""
        ...


class BraveSearch:
    URL = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_key: str | None = None, timeout: float = 10.0):
        self.api_key = api_key or os.environ["PLAGAI_BRAVE_API_KEY"]
        self.client = httpx.Client(timeout=timeout)

    def search(self, query: str, count: int = 5) -> list[dict]:
        r = self.client.get(self.URL, params={"q": query, "count": count},
                            headers={"X-Subscription-Token": self.api_key, "Accept": "application/json"})
        r.raise_for_status()
        return [{"url": x["url"], "title": x.get("title", "")} for x in r.json().get("web", {}).get("results", [])]


def html_to_text(raw: str) -> str:
    raw = _SCRIPT.sub(" ", raw)
    raw = _BLOCK.sub("\n", raw)
    text = html.unescape(_TAG.sub(" ", raw))
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if len(ln.split()) >= 4)


def distinctive_queries(text: str, n: int = 6) -> list[str]:
    """Choose long, information-dense sentences as search queries, spread across the document."""
    sents = [s.text for s in split_sentences(text) if 10 <= len(words(s.text)) <= 40]
    if not sents:
        return []
    step = max(1, len(sents) // n)
    picked = sorted(sents[::step], key=lambda s: -len(set(words(s))))[:n]
    return [f'"{" ".join(s.split()[:20])}"' if i % 2 == 0 else " ".join(s.split()[:25])
            for i, s in enumerate(picked)]


class WebSourceFinder:
    def __init__(self, provider: SearchProvider, encoder: Encoder, max_pages: int = 15,
                 timeout: float = 8.0, max_bytes: int = 2_000_000):
        self.provider = provider
        self.encoder = encoder
        self.max_pages = max_pages
        self.client = httpx.Client(timeout=timeout, follow_redirects=True,
                                   headers={"User-Agent": "plagai-bot/0.1 (+plagiarism check)"})
        self.max_bytes = max_bytes

    def _fetch(self, url: str) -> str:
        try:
            r = self.client.get(url)
            if r.status_code != 200 or "html" not in r.headers.get("content-type", ""):
                return ""
            return html_to_text(r.text[: self.max_bytes])
        except httpx.HTTPError as e:
            log.debug("fetch failed %s: %s", url, e)
            return ""

    def build_index(self, text: str) -> CorpusIndex:
        seen: dict[str, str] = {}
        for q in distinctive_queries(text):
            try:
                for res in self.provider.search(q):
                    seen.setdefault(res["url"], res["title"])
            except httpx.HTTPError as e:
                log.warning("search failed: %s", e)
            if len(seen) >= self.max_pages:
                break
        urls = list(seen)[: self.max_pages]
        with ThreadPoolExecutor(8) as ex:
            pages = list(ex.map(self._fetch, urls))
        index = CorpusIndex(self.encoder)
        for url, page in zip(urls, pages):
            if page:
                index.add(page, title=seen[url], url=url, doc_id=url)
        return index
