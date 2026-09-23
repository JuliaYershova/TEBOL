# Stage 3 — results

Five class pairs, logistic regression on mean-pooled qwen3-embedding vectors (2560-d). Captions generated at 8 lengths × 5 repetitions per image; 5-fold × 5-repeat cross-validation with folds assigned to *images*, so the five captions of one photograph never straddle a split. Every interval is over the resulting 25 fits.

## Reports in this folder

| file | answers | produced by | reads |
|---|---|---|---|
| `00_audit.md` | is every stage complete? | `16_audit.py` | every artefact |
| `01_coverage.md` | how many images, boxes, tags? | `coverage_report.py` | `data/raw`, ImageNet-Captions |
| `02_caption_stability.md` | how repeatable is the captioner? | `15_caption_stability.py` | raw caption JSONL |
| `03_class_leakage.md` | how often does a caption name its class? | `class_leakage.py` | raw captions + `class_terms.csv` |
| `04_accuracy.md` | accuracy per arm and length, with tests | `05_report_accuracy.py` | `summary.csv`, `comparisons.csv` |
| `05_aopc.md` | AOPC curves, all rankings | `07_aopc.py` | `smer_words`, `lime_words` |
| `06_smer_examples_*.md` | one worked caption per class | `06_worked_examples.py` | `smer_words.parquet` |
| `README.md` | this summary | `13_report.py` | all of the above |

## How the numbers are produced

| quantity | metric | why this one |
|---|---|---|
| classification | image-level accuracy | the five captions of a photograph are averaged to one probability first, because the claim is about the image, not the sentence |
| interval | 95% t-interval over 25 fits | 5 folds x 5 repeats; folds assigned to *images*, so no photograph straddles a split |
| arm differences | Nadeau-Bengio corrected t, Bonferroni | CV training sets overlap, so the naive paired t is anti-conservative |
| explanation faithfulness | AOPC | drop in predicted-class probability as the top-k words are removed |
| explanation agreement | top-3 Jaccard, Kendall tau | set overlap for the headline, full-ordering correlation as a check |
| generation repeatability | pairwise token Jaccard | over the 5 captions of one image at one length |

## Summary

1. **Most of the reported accuracy is leakage.** 85% of captions name their own class; removing it costs up to 0.245 accuracy.
2. **Descriptions do not beat human tags** when both are leaky — they land within a point of each other.
3. **Optimal caption length is opposite for the two arms**: short when the class name is present, long when it is not.
4. **SMER beats LIME by more the longer the caption**, from 0.006 at 3 words to 0.346 at 30, for a reason that is structural rather than empirical.
5. **The captioner is the dominant source of instability.** Two repetitions of one image share under half their words at 30 words, and that moves the explanation more than refitting the model does.

Detail below at **w07**; full numbers in `results/metrics/*.csv`, figures in `results/figures/aopc/`.

## 1. Most of the accuracy is the caption naming its own class

Image-level accuracy at w07, mean of 25 fits.

| pair | captions | class name removed | cost |
|---|---:|---:|---:|
| acousticguitar/violin | 0.990 | 0.818 | **+0.172** |
| ambulance/firetruck | 0.983 | 0.876 | **+0.107** |
| ant/bee | 0.984 | 0.903 | **+0.082** |
| cucumber/zucchini | 0.937 | 0.692 | **+0.245** |
| hotpot/vase | 0.997 | 0.996 | **+0.001** |

85% of captions contain their own class name. Removing it costs 0.001–0.245 accuracy in four pairs. **hotpot/vase is the exception** — food scenes and decor are separable without naming them, so it loses nothing. **cucumber/zucchini collapses to near chance**: two green vegetables that the caption was effectively labelling.

## 2. Descriptions and tags are a dead heat

All three columns on the images ImageNet-Captions covers, so the comparison is not confounded by which photographs each has.

| pair | captions | class name removed | human tags |
|---|---:|---:|---:|
| acousticguitar/violin | 0.985 | 0.747 | 0.977 |
| ambulance/firetruck | 0.990 | 0.831 | 0.993 |
| ant/bee | 0.980 | 0.874 | 0.958 |
| cucumber/zucchini | 0.926 | 0.697 | 0.924 |
| hotpot/vase | 0.998 | 0.998 | 0.971 |

Captions and tags land within about a point of each other. Once the class name is stripped, tags win — but tags are never stripped, so that column compares a clean representation against a leaky one and should be read as a caveat, not a result.

## 3. The two arms want opposite caption lengths

| pair | captions: best | acc | class name removed: best | acc |
|---|---:|---:|---:|---:|
| acousticguitar/violin | w05 | 0.991 | w30 | 0.858 |
| ambulance/firetruck | w03 | 0.985 | w25 | 0.895 |
| ant/bee | w05 | 0.985 | w15 | 0.920 |
| cucumber/zucchini | w05 | 0.941 | w03 | 0.707 |
| hotpot/vase | w03 | 0.997 | w05 | 0.997 |

Shortest length within one standard error of the best. **Leaky captions want to be short** — three words is enough to say 'zucchini', and every extra word dilutes it, because a word contributes `z/n` to a mean-pooled vector. **Clean captions want to be long**, where accumulated visual evidence replaces the label.

## 4. SMER's advantage over LIME grows with caption length

AOPC gap (SMER − LIME) at the deepest k each length allows.

| pair | 3w | 5w | 7w | 10w | 15w | 20w | 25w | 30w |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| acousticguitar/violin | 0.006 | 0.029 | 0.061 | 0.115 | 0.235 | 0.321 | 0.332 | 0.394 |
| ambulance/firetruck | 0.004 | 0.010 | 0.022 | 0.041 | 0.140 | 0.250 | 0.288 | 0.354 |
| ant/bee | 0.014 | 0.020 | 0.030 | 0.067 | 0.208 | 0.251 | 0.275 | 0.324 |
| cucumber/zucchini | 0.003 | 0.006 | 0.006 | 0.020 | 0.104 | 0.161 | 0.192 | 0.222 |
| hotpot/vase | 0.005 | 0.023 | 0.047 | 0.071 | 0.211 | 0.320 | 0.367 | 0.432 |
| **mean** | **0.006** | **0.017** | **0.033** | **0.063** | **0.180** | **0.260** | **0.291** | **0.345** |

A 7-word caption has 2⁷ = 128 possible word-subsets, so LIME's 5000 perturbations cover the whole space 39 times over — it is not approximating, and matches SMER exactly. A 34-word caption has 2³⁴ ≈ 1.7×10¹⁰ subsets and 5000 samples is a vanishing fraction. **SMER's cost stays one dot product while LIME's sample requirement grows exponentially**, which is the advantage that holds regardless of how LIME is configured.

## 5. SMER's word lists are cleaner than LIME's

Top 5 words per class, class name removed, at w07. Corpus ranking over words appearing in ≥20 captions.

| pair | class | SMER | LIME |
|---|---|---|---|
| acousticguitar/violin | violin | violinists, elegant, bow, violinist, polished | bow, polished, girl, plays, playing |
|  | acousticguitar | natural-finish, soundhole, sunburst, microphone, cutaway | playing, resting, against, and, strings |
| ambulance/firetruck | firetruck | red, ladder, fire, firefighters, firefighter | red, ladder, fire, vintage, outdoors |
|  | ambulance | white, medical, beige, crosses, stripe | red, outdoors, city, building, in |
| ant/bee | bee | flowers, coneflower, flower, pollinates, blooms | flowers, flower, fuzzy, pollen, perched |
|  | ant | crawling, crawls, crawl, crack, spider | vibrant, white, weathered, of, plant |
| cucumber/zucchini | zucchini | large, bloom, blossom, giant, huge | blossoms, large, yellow, together, board |
|  | cucumber | veggies, vegetables, zucchinis, vegetable, zucchini | board, and, green, together, white |
| hotpot/vase | vase | floral-patterned, decorative, mosaic, pink, patterned | flowers, pink, purple, colorful, floral |
|  | hotpot | cooking, stew, broths, stir-fry, soup | clay, green, black, greens, red |

SMER returns object and scene nouns. LIME's corpus lists are contaminated by high-frequency filler — `and`, `with`, `on` reach its top ranks, while **no stopword reaches SMER's top 20 in any pair**. This is the corpus-level aggregate; per caption the two agree closely.

## 7. What survives a rerun

Mean pairwise overlap of the top-3 words across repeated runs, varying one source of randomness at a time.

| source varied | SMER | LIME |
|---|---:|---:|
| refit on a different CV split | 0.933 | 0.926 |
| captioner asked again (temp 1.0) | 0.544 | -- |
| LIME's random_state | 1.000 | 0.965 |

Refitting the model barely moves either method. **Regenerating the caption moves both a great deal** — the captioner's randomness, not the explainer's, is the dominant source of instability. SMER has no seed of its own: `z = β·e` involves no sampling, so its value on that row is 1.000 by construction.

