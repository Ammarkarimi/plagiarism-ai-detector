# PlagAI: AI-text and plagiarism detection API

A sellable detection service with three parts:

1. **AI-generated text detector.** A fine-tuned DeBERTa-v3 classifier,
   optionally stacked with the Binoculars zero-shot score and stylometric
   features, then calibrated to a target false-positive rate.
2. **Plagiarism detector.** Winnowing fingerprints and MinHash-LSH catch
   verbatim and near-verbatim copying; sentence embeddings catch paraphrase.
   Matches are aligned at sentence level against a public corpus, each
   customer's private corpus and (optionally) the web.
3. **Credit-metered REST API** with API keys, an append-only credit ledger,
   usage logs, rate limiting and automatic refunds on failure.

See [`docs/RESEARCH.md`](docs/RESEARCH.md) for the literature review behind the design: how
others trained these detectors, which datasets exist and why the choices below
were made.

```
src/plagai/
  text.py                  normalisation (homoglyph/zero-width undo), sentence splitting
  ai/data.py               dataset builder: HC3, MAGE, RAID, GPT-wiki-intro, AI-pile (+ your CSVs)
  ai/augment.py            RAID-style adversarial augmentation for training
  ai/train.py              fine-tune the classifier (HF Trainer)
  ai/zero_shot.py          Binoculars zero-shot scorer
  ai/stylometry.py         handcrafted features
  ai/ensemble.py           stacker + threshold calibration at target FPR
  ai/detector.py           inference: document label + per-segment highlights
  plagiarism/fingerprint.py  winnowing, MinHash, LSH
  plagiarism/index.py      corpus index (fingerprints + LSH + embeddings), persistence
  plagiarism/detector.py   alignment, passage merging, report
  plagiarism/web.py        web-source finder (Brave Search API, pluggable)
  plagiarism/train_paraphrase.py  fine-tune encoder on PAWS + MRPC
  api/                     FastAPI app, DB models, billing, rate limiting
```

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu130  # NVIDIA GPU (driver 580+); /whl/cu126 for older drivers, /whl/cpu for no GPU
pip install -e ".[dev]"
pytest                                   # 23 tests, no downloads needed
```

## 1. Train the AI detector

```bash
# 1a. Build the dataset (samples random Parquet row groups; doesn't download all of RAID)
plagai prepare-ai-data --out data/ai --sources hc3 mage raid wiki ai_pile \
    --per-source 40000 --ood-source wiki            # wiki is held out to measure generalisation
#   add Kaggle DAIGT-v2 or your own data:  --extra-csv path/to/daigt.csv  (columns: text,label with 1 = AI)

# 1b. Fine-tune (one A100/L4/4090: about 2-5 h)
plagai train-ai --data data/ai --model microsoft/deberta-v3-base --out artifacts/ai-detector \
    --epochs 2 --batch-size 16 --max-length 512 --lr 2e-5 --bf16

#     6-8 GB GPU (e.g. RTX 3050/3060 laptop): same effective batch, about 4.2 GB VRAM,
#     about 9-10 h for 110k texts x 2 epochs
plagai train-ai --data data/ai --model microsoft/deberta-v3-base --out artifacts/ai-detector \
    --epochs 2 --batch-size 4 --grad-accum 4 --gradient-checkpointing --max-length 512 --lr 2e-5 --bf16 \
    --max-eval 3000

# 1c. Fit the stacker and calibrate thresholds (1% FPR on human text by default)
plagai fit-ensemble --model-dir artifacts/ai-detector --data data/ai            # add --binoculars for zero-shot features
```

Training notes:
* `--bf16`/`--fp16` fall back to what your hardware supports. If the log says `device=cpu`, you
  have the CPU build of torch; reinstall it from the CUDA index above.
* Before training, the script checks there's enough free disk for checkpoints. Afterwards it deletes
  the checkpoints; the final model (about 0.7 GB) is in `--out`. Pass `--keep-checkpoints` or
  `--save-optimizer` if you need to resume.
* Stylometric features are opt-in (`fit-ensemble --stylometry`). In our tests they lowered
  accuracy on unseen sources.

Outputs in `artifacts/ai-detector/`:
* `classifier_metrics.json`: AUROC and TPR@1%FPR per source, generator and attack
* `ensemble_metrics.json`: the same for the final calibrated ensemble

Check `test_ood` before you ship.

Scaling tips:
* Use `microsoft/deberta-v3-large` with `--gradient-checkpointing --batch-size 8 --grad-accum 4` for about 2-4 more AUROC points.
* Add more generators. Fresh outputs from current frontier models (GPT, Claude,
  Gemini, Llama, Qwen, DeepSeek) on *your customers' domains* (student essays,
  marketing copy and so on) are the biggest accuracy lever. Write them to a CSV
  and pass it with `--extra-csv`.
* Retrain quarterly. Detectors decay as new models ship.

## 2. Plagiarism corpus and encoder

```bash
# Optional: fine-tune the paraphrase encoder (prints a recommended cosine threshold)
plagai train-paraphrase --base sentence-transformers/all-MiniLM-L6-v2 --out artifacts/paraphrase-encoder
export PLAGAI_EMBEDDING_MODEL=artifacts/paraphrase-encoder

# Index a shared reference corpus: a folder of .txt files, or JSONL with {"text","title","url"}
plagai index-corpus --input refs/ --namespace public
```

Customers add their own documents (for example, past student submissions)
with `POST /v1/documents`. Those go into a private namespace that only their
account can search. Web search turns on when you set `PLAGAI_BRAVE_API_KEY`.

## 3. Run the API

```bash
cp .env.example .env               # set PLAGAI_ADMIN_TOKEN!
plagai serve --port 8000           # or: docker compose up
```

Create a customer, sell them credits and hand them a key:

```bash
curl -X POST localhost:8000/admin/accounts -H "X-Admin-Token: $TOKEN" \
     -H 'content-type: application/json' -d '{"email":"buyer@example.com","credits":1000}'
# => {"account_id": 1, "api_key": "plg_live_...", ...}   (the key is shown only once)

curl -X POST localhost:8000/admin/accounts/1/credits -H "X-Admin-Token: $TOKEN" \
     -H 'content-type: application/json' -d '{"amount":5000,"reason":"purchase","reference":"stripe_pi_123"}'
```

Or use the CLI: `plagai admin create-account buyer@example.com --credits 1000`.

Customer usage:

```bash
curl -X POST localhost:8000/v1/ai-detect -H "Authorization: Bearer plg_live_..." \
     -H 'content-type: application/json' -d '{"text": "..."}'
curl -X POST localhost:8000/v1/plagiarism/check -H "Authorization: Bearer plg_live_..." \
     -H 'content-type: application/json' -d '{"text": "...", "scope": "all", "web": false}'
```

| Endpoint | Purpose | Cost (default) |
|---|---|---|
| `POST /v1/ai-detect` | AI probability, label (`ai` / `human` / `mixed` / `uncertain`) and per-segment scores | 1 credit per 100 words |
| `POST /v1/plagiarism/check` | Similarity %, matched passages, sources | 1 per 100 words (3× with `web: true`) |
| `POST /v1/analyze` | Both checks together | 2 per 100 words |
| `POST /v1/documents`, `DELETE /v1/documents/{id}` | Manage your private reference corpus | 0.2 per 100 words |
| `GET /v1/account`, `GET /v1/usage` | Balance and request history | free |

The minimum charge is 1 credit per call. Every response includes `usage`
(credits charged and remaining) and an `X-Credits-Remaining` header. If an
analysis fails, its credits are refunded automatically. Interactive docs are
at `/docs`.

### Selling it

* **Payments:** point your Stripe (or Paddle or Lemon Squeezy) `checkout.session.completed`
  webhook handler at `POST /admin/accounts/{id}/credits` with
  `reference = payment id`. Grants with the same reference are applied only
  once, so webhook retries are safe.
* **Before production:** use Postgres (`PLAGAI_DATABASE_URL`), put the API
  behind TLS and a gateway, move rate limiting to Redis when you run several
  replicas, and run the model on a GPU worker if latency matters.
* **Responsible use:** no detector is perfect. Display the
  `uncertain`/`mixed` bands, avoid presenting results as proof, and warn about
  texts under 80 words and non-native English writing. The API returns
  `warnings` for short texts.

## Smoke-test results (CPU, tiny model, for pipeline validation only)

See [`docs/SMOKE_TEST.md`](docs/SMOKE_TEST.md). These numbers come from a
14M-parameter model trained for about 3 minutes on 3k texts. They show that
the pipeline works end to end. They are **not** production accuracy.
