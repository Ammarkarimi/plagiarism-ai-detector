import numpy as np

from plagai.plagiarism.detector import PlagiarismDetector, containment
from plagai.plagiarism.embeddings import HashingEncoder
from plagai.plagiarism.fingerprint import MinHasher, fingerprints, kgram_hashes, winnow
from plagai.plagiarism.index import CorpusIndex
from plagai.text import words

SOURCE = (
    "The mitochondrion is the powerhouse of the cell and produces most of its chemical energy. "
    "It generates adenosine triphosphate through a process called oxidative phosphorylation. "
    "Mitochondria have their own DNA, which is inherited exclusively from the mother in most species. "
    "Scientists believe they originated from free-living bacteria engulfed by ancestral cells."
)
OTHER = (
    "Volcanoes form where tectonic plates diverge or converge and magma reaches the surface. "
    "Some eruptions are explosive while others release slow rivers of basaltic lava across the land."
)


def _index():
    idx = CorpusIndex(HashingEncoder())
    idx.add(SOURCE, title="Biology 101", url="https://example.org/bio", doc_id="bio")
    idx.add(OTHER, title="Geology", doc_id="geo")
    return idx


def test_winnowing_guarantee():
    toks = words(SOURCE)
    fps = {h for h, _ in fingerprints(toks, 5, 4)}
    # Any copied run of at least window + k - 1 = 8 words shares a fingerprint.
    copied = toks[10:20]
    assert {h for h, _ in winnow(kgram_hashes(copied, 5), 4)} & fps


def test_minhash_estimates_jaccard():
    mh = MinHasher(256)
    a = kgram_hashes(words(SOURCE), 3)
    assert MinHasher.jaccard(mh.signature(a), mh.signature(a)) == 1.0
    assert MinHasher.jaccard(mh.signature(a), mh.signature(kgram_hashes(words(OTHER), 3))) < 0.1


def test_containment():
    assert containment("the cat sat on the mat", "yesterday the cat sat on the mat quietly") == 1.0
    assert containment("dogs bark loudly at night", "the cat sat on the mat") == 0.0


def test_detects_verbatim_copy_and_ignores_original_text():
    idx = _index()
    det = PlagiarismDetector(HashingEncoder())
    essay = ("In my essay I want to discuss biology. "
             "It generates adenosine triphosphate through a process called oxidative phosphorylation. "
             "Mitochondria have their own DNA, which is inherited exclusively from the mother in most species. "
             "My grandmother grew tomatoes in a small garden behind her house every summer.")
    rep = det.check(essay, [("public", idx)])
    assert rep["sources"][0]["doc_id"] == "bio"
    assert len(rep["passages"]) == 1 and rep["passages"][0]["type"] == "exact"
    assert "tomatoes" not in rep["passages"][0]["text"]
    assert 30 < rep["similarity_percent"] < 80

    clean = det.check("My grandmother grew tomatoes in a small garden behind her house every summer. "
                      "She sold the extras at the farmers market on Saturday mornings.", [("public", idx)])
    assert clean["label"] == "original" and clean["passages"] == []


def test_near_duplicate_detection():
    idx = _index()
    rep = PlagiarismDetector(HashingEncoder()).check(SOURCE.replace("most", "many"), [("public", idx)])
    assert rep["near_duplicates"][0]["doc_id"] == "bio"
    assert rep["label"] == "high"


def test_index_persistence_and_removal(tmp_path):
    idx = _index()
    idx.save(tmp_path)
    loaded = CorpusIndex.load(tmp_path, HashingEncoder())
    assert set(loaded.docs) == {"bio", "geo"}
    assert np.allclose(loaded.embeddings, idx.embeddings)
    assert loaded.remove("bio")
    rep = PlagiarismDetector(HashingEncoder()).check(SOURCE, [("public", loaded)])
    assert rep["passages"] == []
