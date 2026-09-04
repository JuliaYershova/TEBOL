# TEBOL

Explaining an image classifier that reads generated captions rather than pixels.
A vision model describes each photograph, the words are embedded, logistic
regression classifies the mean-pooled vector, and SMER decomposes that
prediction back onto the words. LIME is the comparison; AOPC is the test.

**Five binary pairs** — acousticguitar/violin, ambulance/firetruck, ant/bee,
cucumber/zucchini, hotpot/vase. **574,040 captions** per variant: every image
described at 8 lengths (3–30 words) × 5 repetitions.

Every number below traces to a script and a report. Start with
[`results/reports/00_audit.md`](results/reports/00_audit.md) — 36 checks over
the whole pipeline — and [`results/reports/README.md`](results/reports/README.md)
for the findings.

## Pipeline

| # | script | what it does | output |
|---|---|---|---|
| 1 | `01_caption.py` | asks the VLM for a caption, 8 lengths × 5 reps per image | `artifacts/captions/` |
| — | `class_leakage.py` | measures how often a caption names its own class, and writes the class-stripped twin of every caption | `--noclass.jsonl`, [`03_class_leakage.md`](results/reports/03_class_leakage.md) |
| — | `build_caption_tags.py` | pulls the human Flickr tags from ImageNet-Captions | `data/manifest/tags/` |
| 2 | `02_embed.py` | embeds each distinct word once, then joins captions to vocabulary rows | `artifacts/embeddings/` |
| 3 | `03_train_smer.py` | fits logistic regression under 5-fold × 5-repeat CV and decomposes it into per-word logits | `coefs.npz`, `oof.parquet`, `smer_words.parquet` |
| 4 | `09_lime.py` | LIME on the same captions, for comparison | `lime_words*.parquet` |
| 5 | `07_aopc.py` | AOPC curves for every ranking | `aopc.csv`, [`05_aopc.md`](results/reports/05_aopc.md) |
| 6 | `04_compare_arms.py` | paired significance tests between arms | `comparisons.csv` |
| 7 | `10_figure_sets.py` | 205 figures in six comparison folders | `results/figures/aopc/` |
| 8 | `11/12_*.py`, `15_caption_stability.py` | how much survives a rerun | [`02_caption_stability.md`](results/reports/02_caption_stability.md) |
| 9 | `13_report.py`, `16_audit.py` | the summary and the completeness audit | [`README`](results/reports/README.md), [`audit`](results/reports/00_audit.md) |

`06_worked_examples.py`, `05_report_accuracy.py` and `14_caption_length.py`
produce supporting tables; `coverage_report.py` counts the raw data.

## The five arms

What text the classifier sees. They separate effects the naive setup conflates.

| arm | text | scope |
|---|---|---|
| `caption` | descriptions as generated | all captioned images |
| `caption_noclass` | the same, class name removed | all captioned images |
| `tags` | human ImageNet-Captions tags | tag-covered images |
| `caption_tagsub` | descriptions | tag-covered images |
| `caption_noclass_tagsub` | descriptions, class name removed | tag-covered images |

Any figure containing a tag line uses the tag-covered subset for *every* line —
tags exist for about a third of the corpus, and mixing scopes would confound
representation with sample.

## What was found

1. **Most reported accuracy is leakage.** 85% of captions name their own class;
   removing it costs up to 0.245 accuracy — except hotpot/vase, which loses
   nothing.
2. **Descriptions do not beat human tags** when both are leaky.
3. **Caption length trades one thing for another**: accuracy peaks at ~5 words
   with the class name present, 15–30 without it.
4. **SMER's advantage over LIME grows with length**, 0.006 at 3 words to 0.346
   at 30 — LIME's perturbation budget cannot cover a 2³⁴ subset space.
5. **The captioner is the dominant source of instability**, not the model and
   not the explainer.

Detail in [`results/reports/README.md`](results/reports/README.md).

## Layout

```
artifacts/   captions, embeddings, models, explanations   (gitignored, large)
data/        raw images, manifests, ImageNet-Captions
results/
  reports/   9 markdown reports, numbered in pipeline order
  metrics/   the CSVs behind them
  figures/   205 AOPC figures in 6 folders + caption length
scripts/     the pipeline, numbered
src/tebol/   the library the scripts call
```

## Running it

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env            # gateway URL, key, model names
.venv/bin/python scripts/16_audit.py
```

The audit says what is already complete. Stages 1 and 2 need the inference
gateway; 3 onwards are local. `03_train_smer.py` takes ~7 minutes for all 165
configurations, `09_lime.py` ~25, `07_aopc.py` ~10.

## Not done

Bounding boxes. `src/tebol/bbox/` is empty; ground truth exists for 4,531
images (see [`01_coverage.md`](results/reports/01_coverage.md)).
