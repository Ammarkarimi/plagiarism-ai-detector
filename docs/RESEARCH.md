# Research notes: how AI-text and plagiarism detectors are built

These notes record what the published work shows and how it shaped the design
of this repository. Read them before you change the training recipe.

---

## 1. AI-generated text detection

Detectors fall into three families. The strongest commercial systems combine
all three.

### 1.1 Supervised classifiers (fine-tuned encoders)

| Work | What they did | Takeaway |
|---|---|---|
| OpenAI GPT-2 output detector (Solaiman et al., 2019) | Fine-tuned RoBERTa on GPT-2 vs WebText | Strong in-distribution, but weak on new generators |
| HC3 / "How close is ChatGPT to human experts?" (Guo et al., 2023) | RoBERTa fine-tuned on human vs ChatGPT answers | Released the **HC3** dataset |
| MAGE / "DeepfakeTextDetect" (Li et al., 2024) | Longformer on 27 LLMs × 7 domains | Out-of-distribution (unseen-model/unseen-domain) splits are what matter |
| M4 / M4GT-Bench (Wang et al., 2024) | Multi-generator, multi-domain, multilingual | Cross-generator transfer drops sharply |
| Kaggle *LLM - Detect AI Generated Text* (2023–24) | Winners: **DeBERTa-v3** + TF-IDF on a diverse synthetic set (DAIGT v2), heavy augmentation (typos, character noise) | **Data diversity beats model size.** Augmentation against obfuscation matters. |
| RADAR (Hu et al., 2023) | Detector trained adversarially against a paraphraser | Paraphrase-robustness has to be trained in |
| GenAI Content Detection Task 1 (COLING 2025) | Training data merged from HC3 + M4GT + MAGE; top systems were fine-tuned encoders and ensembles | The same multi-source recipe we use |

### 1.2 Zero-shot / statistical detectors (no training)

| Method | Signal |
|---|---|
| GLTR (2019) | Token rank histogram under a reference LM |
| Perplexity + "burstiness" (GPTZero) | Low, even perplexity across sentences suggests AI text |
| DetectGPT (Mitchell et al., 2023) | Probability curvature under perturbations (slow) |
| Fast-DetectGPT (Bao et al., 2024) | Analytic conditional-probability curvature, about 340× faster than DetectGPT |
| **Binoculars** (Hans et al., ICML 2024) | Perplexity / cross-perplexity between two sibling LMs. Over 90% TPR at 0.01% FPR on ChatGPT text with **no training** |

Zero-shot methods generalise to new generators better than classifiers. But
they are sensitive to domain, and they misfire on non-native English
(Liang et al., 2023, "GPT detectors are biased against non-native English
writers").

### 1.3 Benchmarks that expose weaknesses

* **RAID** (Dugan et al., ACL 2024): 6M+ generations from 11 LLMs × 8 domains
  × 11 adversarial attacks (homoglyphs, paraphrase, whitespace, synonym swaps
  and others) × 4 decoding strategies. Most detectors collapse at low FPR under
  attacks or with repetition-penalty sampling.
* **DetectRL** (2024): real-world attacks; fine-tuned RoBERTa beat zero-shot
  methods in-domain.
* Stacking zero-shot scores and a classifier in a meta-model cuts the FPR on
  RAID substantially (community results, not peer reviewed).

### 1.4 Design decisions that follow

1. **The primary model is a fine-tuned DeBERTa-v3** (`microsoft/deberta-v3-base`,
   or `-large` if you have the GPU budget). It is trained on a **multi-source
   mix**: HC3 + MAGE + RAID (train, including the adversarial attacks) +
   GPT-wiki-intro + ai-text-detection-pile. Sources are balanced, so no single
   generator or domain dominates.
2. **Adversarial augmentation at training time** covers homoglyphs, zero-width
   characters, typos, whitespace noise, casing and partial paraphrase. These
   mirror the RAID attacks. The API also *normalises* homoglyphs and zero-width
   characters before scoring.
3. **A Binoculars zero-shot score** (two sibling causal LMs sharing a
   tokenizer) is a second, training-free signal.
4. **Stylometric features** (sentence-length burstiness, type-token ratio,
   punctuation profile and others) are cheap and add complementary signal.
5. **A logistic-regression meta-classifier** (stacker) combines the signals.
   It is fitted on a held-out split, never on the classifier's own training
   data.
6. **Threshold at a target FPR** (default 1%). A false accusation is the
   costliest error for a paid product, so the decision threshold is picked on
   human validation text, not at 0.5. An "uncertain" band is returned between
   the low and high thresholds.
7. **Evaluate on out-of-distribution splits** (held-out generator, held-out
   domain) and report AUROC **and TPR@1%FPR**. AUROC alone hides the
   low-FPR collapse.
8. **Chunked inference.** Long documents are split into token windows; the API
   returns document-level and per-sentence/segment scores so users can see
   *where* the AI text is.

---

## 2. Plagiarism detection

### 2.1 How others do it

| Stage | Classic approach | Modern approach |
|---|---|---|
| Source retrieval (candidate selection) | Word k-gram **fingerprinting / winnowing** (Schleimer et al., 2003, the MOSS algorithm); **MinHash + LSH** for near-duplicates (Broder, 1997) | Dense retrieval: chunk embeddings (e.g. e5, BGE, MiniLM) in an ANN index |
| Detailed comparison (text alignment) | Seed-and-extend on shared n-grams (PAN text-alignment winners) | Sentence-embedding similarity for paraphrase; cross-encoder re-ranking |
| Paraphrase / AI-rewrite robustness | Synonym normalisation | Embedding models fine-tuned on paraphrase data (PAWS, MRPC, machine-paraphrase corpora) |

* The **PAN shared tasks** (2009–2015, 2025) are the standard benchmark, with
  character-level span annotations. PAN-PC-11 and PAN 2013 include obfuscated,
  translated and paraphrased cases. PAN 2025 targets **LLM-generated
  plagiarism** of scientific papers.
* Hybrid lexical + semantic systems (for example TF-IDF + MiniLM fine-tuned
  on PAN-11, 2025) beat purely lexical ones, especially on paraphrase.
* Wahle et al. (2022) released the **machine-paraphrase** (SpinBot,
  SpinnerChief) and **autoregressive-paraphrase** (GPT-3, T5) datasets for
  detecting machine-paraphrased plagiarism.
* A recent PAN 2025 system chunks documents with a sliding window of 6
  sentences and stride 2, embeds them with e5-base-v2 and aligns the chunks.

### 2.2 Design decisions that follow

1. **Two-stage pipeline:** cheap candidate retrieval, then expensive alignment.
2. Retrieval combines **winnowing fingerprints** (exact and near-verbatim
   copying), **MinHash-LSH** (near-duplicate documents) and **dense
   embeddings** (paraphrase).
3. Alignment is **sentence-level**: each query sentence is matched to its best
   source sentence. The match is labelled *exact*, *near-exact* or
   *paraphrase* from the lexical overlap and cosine similarity. Adjacent
   matches are merged into passages.
4. The embedding model can be **fine-tuned** on PAWS + machine-paraphrase +
   autoregressive-paraphrase with a contrastive loss (`train_paraphrase.py`).
5. Users can add their own reference corpora through the API (for example,
   a university's past submissions). Web search is a pluggable source
   provider, so you can wire in a search API such as Bing, Brave or Google
   Custom Search.

---

## 3. Datasets used (all on the Hugging Face Hub, checked 2026-10)

| Dataset | Hub id | Size | Label convention | License |
|---|---|---|---|---|
| HC3 | `Hello-SimpleAI/HC3` (config `all`) | 24k questions, human vs ChatGPT answers | separate `human_answers` / `chatgpt_answers` lists | CC-BY-SA |
| MAGE | `yaful/MAGE` | 319k train / 57k val / 61k test | **`label=1` means HUMAN**, `0` means machine (inverted in the loader) | Apache-2.0 |
| RAID | `liamdugan/raid` (config `raid`) | 2.27M train | `model == "human"` means human; `attack` column for adversarial variants | MIT |
| GPT-wiki-intro | `aadityaubhat/GPT-wiki-intro` | 150k paired intros | `wiki_intro` is human, `generated_intro` is AI | CC |
| AI text detection pile | `artem9k/ai-text-detection-pile` | 1.39M | `source` is `human` or `ai` | MIT |
| PAWS | `google-research-datasets/paws` (`labeled_final`) | 49k pairs | `label=1` means paraphrase | free use |
| Machine-paraphrase | `jpwahle/machine-paraphrase-dataset` | 200k | `label=1` means spun/paraphrased | CC-BY |
| Autoregressive-paraphrase | `jpwahle/autoregressive-paraphrase-dataset` | 262k | `label=1` means paraphrased | CC-BY |

Not on the Hub, but recommended:
* **DAIGT v2** (Kaggle `thedrcat/daigt-v2-train-dataset`) is student essays
  from many LLMs. Download it manually and pass it with `--extra-csv`.
* **PAN plagiarism corpora** (pan.webis.de) are for *evaluating* the
  plagiarism aligner.
* **M4GT-Bench** (GitHub `mbzuai-nlp/M4GT-Bench`) is multilingual. Use it if
  you sell outside English.

Check each dataset's license before you train a commercial model. Some HC3
sources (Reddit ELI5, WikiQA) carry upstream terms.

---

## Sources

* RAID benchmark: https://arxiv.org/abs/2405.07940
* Binoculars: https://arxiv.org/abs/2401.12070 and https://proceedings.mlr.press/v235/hans24a.html
* DetectRL: https://arxiv.org/abs/2410.23746
* GenAI Content Detection Task 1 (HC3 + M4GT + MAGE): https://arxiv.org/abs/2501.11012
* Hybrid lexical + semantic plagiarism detection (PAN-11 fine-tuned MiniLM): https://www.aseestant.ceon.rs/index.php/jcfs/article/view/64419
* Winnowing for text plagiarism (arXiv): https://arxiv.org/abs/cs/0702012
* PAN 2025 generative plagiarism system: https://ceur-ws.org/Vol-4038/paper_324.pdf
* Fast-DetectGPT: https://arxiv.org/abs/2310.05130
* MAGE: https://arxiv.org/abs/2305.13242
* HC3: https://arxiv.org/abs/2301.07597
