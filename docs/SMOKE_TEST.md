# Smoke test (CPU, 2026-10-09)

**Purpose:** prove the full pipeline runs end to end on real data:
download, train, calibrate, serve and bill. These numbers come from a
14M-parameter model trained for 5.5 minutes on about 3k texts on 4 CPU
cores. They are **not** an estimate of production accuracy. A
DeBERTa-v3 model trained on 200k+ texts on a GPU is the real target.

Reproduce with `scripts/smoke_test.sh`.

## Data (`plagai prepare-ai-data --per-source 1000 --ood-source wiki`)

| split | n | sources |
|---|---|---|
| train | 3,209 | HC3, MAGE, RAID, AI-pile |
| val | 368 | same |
| test | 423 | same (group-disjoint from train) |
| test_ood | 1,000 | GPT-wiki-intro only (source never seen in training) |

The sample covers 8 MAGE domains and 15+ generators (OPT, LLaMA, GPT-3.5,
davinci, Flan-T5, BLOOM and others). From RAID it covers GPT-4, Mistral, MPT,
Cohere and GPT-2, with attacks including homoglyph, paraphrase, whitespace,
upper_lower and article deletion.

## AI detector

Classifier: `google/electra-small-discriminator`, 1 epoch, max length 256, lr 1e-4.

| | AUROC | TPR @ 1% FPR | TPR @ 5% FPR | FPR at decision threshold |
|---|---|---|---|---|
| val (during training) | 0.920 | 0.21 | n/a | n/a |
| test, classifier alone (threshold 0.5) | 0.927 | 0.15 | 0.73 | **33%** |
| test_ood, classifier alone (threshold 0.5) | 0.877 | 0.01 | 0.23 | **73%** |
| test, **ensemble** (calibrated threshold) | 0.936 | **0.56** | 0.78 | **0.0%** |
| test_ood, **ensemble** (calibrated threshold) | 0.874 | **0.24** | 0.50 | **0.5%** |

The ensemble is classifier + Binoculars (gpt2/distilgpt2, a toy pair) +
stylometry, using logistic regression with thresholds calibrated on
out-of-fold validation predictions (`ai` = 0.958, `human` = 0.243). Ensemble
rows were capped at 400 per split, so its test sets are a subset of the
classifier's.

What this shows:
* An uncalibrated 0.5 threshold is unusable for a product: 33–73% of human
  texts would be flagged. Calibrating at a target FPR keeps false accusations
  near 1% while preserving most detection power.
* Stacking zero-shot and stylometric signals raised TPR@1%FPR by 3.7× in-domain
  and about 17× out-of-domain, which matches the literature (docs/RESEARCH.md §1.3).
* OOD performance is the weak spot, as expected from such a small model and
  dataset. This is what full-scale training must fix.

Cost note: Binoculars with full-vocabulary cross-entropy is slow on CPU (about
19 min for 1,200 texts even with gpt2). In production, run it on a GPU or
leave it out of the ensemble.

## Plagiarism

* Encoder fine-tuning (`all-MiniLM-L6-v2`, 2k PAWS+MRPC pairs, 1 min): dev
  cosine accuracy 0.599 → 0.676, AP 0.570 → 0.640, recommended threshold 0.90.
* Base MiniLM cosine: paraphrase pair 0.887, unrelated pair 0.19–0.24.

## API demo (`scripts/demo_api.py`)

```
human-ish  label=human  ai_probability=0.012  (classifier 0.157, binoculars 0.917)   1 credit
ai-ish     label=ai     ai_probability=0.990  (classifier 0.925, binoculars 0.788)   1 credit
plagiarism similarity=67.2%  label=high  exact=1 paraphrase=1                        1 credit
account    96 credits left (100 - 3 checks - 1 document indexing)
```
