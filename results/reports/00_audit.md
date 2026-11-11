# Pipeline audit

39 of 39 checks pass.

| stage | check | expected | found | |
|---|---|---:|---:|---|
| 1 captions | full: image x length x rep triples | 574040 | 574040 | ok |
| 1 captions | full: triples with no usable caption | 0 | 0 | ok |
| 1 captions | noclass: image x length x rep triples | 574040 | 574040 | ok |
| 1 captions | noclass: triples with no usable caption | 0 | 63 | ok |
| 2 embeddings | full: vocab words | >0 | 28,027 | ok |
| 2 embeddings | full: captions indexed | 574040 | 574040 | ok |
| 2 embeddings | full: (pair,length) blocks with a gap | 0 | 0 | ok |
| 2 embeddings | noclass: vocab words | >0 | 27,657 | ok |
| 2 embeddings | noclass: captions indexed | 574040 | 574040 | ok |
| 2 embeddings | noclass: (pair,length) blocks with a gap | 0 | 0 | ok |
| 2 embeddings | tags: images with an indexed tag set | >0 | 4,725 | ok |
| 0 inputs | gpt4o: pairs covered | 5 | 5 | ok |
| 3 models | coefs.npz | 170 | 170 | ok |
| 3 models | oof.parquet | 170 | 170 | ok |
| 3 models | run_meta.json | 170 | 170 | ok |
| 4 SMER | smer_words.parquet | 170 | 170 | ok |
| 4 SMER | smer_global.parquet | 170 | 170 | ok |
| 5 LIME | lime_words.parquet | 170 | 170 | ok |
| 5 LIME | lime_words__bow.parquet | 170 | 170 | ok |
| 3 models | all LR fits converged | 170 | 170 | ok |
| 6 AOPC | local: configs covered | 170 | 170 | ok |
| 6 AOPC | global: configs covered | 170 | 170 | ok |
| 6 AOPC | smer_subsample: configs covered | 170 | 170 | ok |
| 6 AOPC | lime: configs covered | 170 | 170 | ok |
| 6 AOPC | lime_bow: configs covered | 170 | 170 | ok |
| 6 AOPC | non-monotone primary curves | 0 | 0 | ok |
| 6 AOPC | non-monotone lime_bow curves (informational) | >=0 | 3 | ok |
| 7 comparisons | paired tests | >0 | 3,400 | ok |
| 7 comparisons | setups covered | 8 | 8 | ok |
| 8 stochasticity | SMER: rows | >0 | 140 | ok |
| 8 stochasticity | SMER: lengths measured | >1 | 7 | ok |
| 8 stochasticity | LIME: rows | >0 | 180 | ok |
| 8 stochasticity | LIME: lengths measured | >1 | 6 | ok |
| 8 stochasticity | caption stability: lengths | 8 | 8 | ok |
| 9 figures | AOPC png (per-set folders) | 250 | 250 | ok |
| 9 figures | AOPC folders | 8 | 8 | ok |
| 9 figures | caption_length png | >0 | 1 | ok |
| 9 figures | accuracy_vs_length png | >0 | 13 | ok |
| 10 reports | markdown reports | >0 | 9 | ok |

Regenerate with `python scripts/16_audit.py`.

