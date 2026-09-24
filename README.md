# TEBOL

Explaining an image classifier that reads generated captions rather than pixels.
Five binary pairs: acousticguitar/violin, ambulance/firetruck, ant/bee,
cucumber/zucchini, hotpot/vase.

## Pipeline

| # | step | script |
|---|---|---|
| 1 | A vision model describes each photograph, 8 lengths × 5 repetitions. | `01_caption.py` |
| 2 | Every distinct word is embedded once; a caption is the mean of its words. | `02_embed.py` |
| 3 | Logistic regression classifies that vector, 5-fold CV × 5 repeats. | `03_train_smer.py` |
| 4 | SMER decomposes each prediction back onto the individual words. | `03_train_smer.py` |
| 5 | LIME explains the same captions, as the comparison. | `09_lime.py` |
| 6 | AOPC tests both rankings by deleting their top words. | `07_aopc.py` |
| 7 | The top-3 words are given to the vision model, which returns a box for them. | `23_bbox.py` |

## Evaluation baselines

Each baseline removes one part of the pipeline, so the accuracy it loses is
what that part contributes. All are scored on the same images, the same
5 × 5 folds and the same image-level rule.

| baseline | what it removes |
|---|---|
| **LR no-class** | the class word, deleted from every caption |
| **name in text** | the classifier — predict whichever class the caption names |
| **zero-shot** | the captions and the training; the VLM is asked to classify directly |
| **CLIP / CoCa** | the same, with a contrastive model and the class names in the prompt |
| **CLIP / CoCa probe** | the captions only — the same classifier on frozen image vectors, never shown a class name |

Accuracy at 5-word captions, image level, ± is the 95% half-width:

| pair | LR caption | LR no-class | name in text | zero-shot | CLIP | CLIP probe | CoCa | CoCa probe |
|---|---|---|---|---|---|---|---|---|
| acousticguitar/violin | **0.9907** ±.001 | 0.8177 ±.006 | 0.8440 ±.012 | 0.9848 ±.004 | 0.9737 ±.005 | 0.9822 ±.002 | 0.9788 ±.005 | 0.9830 ±.002 |
| ambulance/firetruck | 0.9852 ±.002 | 0.8794 ±.007 | 0.9225 ±.010 | 0.9853 ±.004 | 0.9822 ±.005 | **0.9861** ±.002 | 0.9787 ±.005 | 0.9788 ±.002 |
| ant/bee | **0.9849** ±.002 | 0.8868 ±.005 | 0.8975 ±.010 | 0.9772 ±.005 | 0.9426 ±.008 | 0.9689 ±.003 | 0.9384 ±.008 | 0.9606 ±.003 |
| cucumber/zucchini | **0.9410** ±.004 | 0.7087 ±.010 | 0.8191 ±.015 | 0.8882 ±.013 | 0.6709 ±.019 | 0.8242 ±.005 | 0.7150 ±.018 | 0.8240 ±.005 |
| hotpot/vase | 0.9976 ±.001 | 0.9967 ±.001 | 0.6460 ±.019 | 0.9980 ±.002 | 0.9951 ±.003 | **0.9992** ±.001 | 0.9980 ±.002 | 0.9988 ±.001 |

Intervals are not the same kind and should not be compared as widths: a
t-interval over 25 folds where a model is refit, a bootstrap over images where
it is not. Paired differences and significance are in
[`results/metrics/baselines_w05.csv`](results/metrics/baselines_w05.csv).

```
python scripts/24_baselines.py run --pair ant_bee --baseline clip
python scripts/24_baselines.py table
```
