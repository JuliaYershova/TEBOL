# Pipeline audit

36 of 36 checks pass.

| stage | check | expected | found | |
|---|---|---:|---:|---|
| 1 captions | full: image x length x rep triples | 574040 | 574040 | ok |
| 1 captions | full: triples with no usable caption | 0 | 0 | ok |
| 1 captions | noclass: image x length x rep triples | 574040 | 574040 | ok |
| 1 captions | noclass: triples with no usable caption | 0 | 14 | ok |
| 2 embeddings | full: vocab words | >0 | 28,027 | ok |
| 2 embeddings | full: captions indexed | 574040 | 574040 | ok |
| 2 embeddings | full: (pair,length) blocks with a gap | 0 | 0 | ok |
| 2 embeddings | noclass: vocab words | >0 | 27,657 | ok |
| 2 embeddings | noclass: captions indexed | 574040 | 574040 | ok |
| 2 embeddings | noclass: (pair,length) blocks with a gap | 0 | 0 | ok |
| 2 embeddings | tags: images with an indexed tag set | >0 | 4,725 | ok |
| 3 models | coefs.npz | 165 | 165 | ok |
| 3 models | oof.parquet | 165 | 165 | ok |
| 3 models | run_meta.json | 165 | 165 | ok |
| 4 SMER | smer_words.parquet | 165 | 165 | ok |
| 4 SMER | smer_global.parquet | 165 | 165 | ok |
| 5 LIME | lime_words.parquet | 165 | 165 | ok |
| 5 LIME | lime_words__bow.parquet | 165 | 165 | ok |
| 3 models | all LR fits converged | 165 | 165 | ok |
| 6 AOPC | local: configs covered | 165 | 165 | ok |
| 6 AOPC | global: configs covered | 165 | 165 | ok |
| 6 AOPC | smer_subsample: configs covered | 165 | 165 | ok |
| 6 AOPC | lime: configs covered | 165 | 165 | ok |
| 6 AOPC | lime_bow: configs covered | 165 | 165 | ok |
| 6 AOPC | non-monotone primary curves | 0 | 0 | ok |
| 7 comparisons | paired tests | >0 | 3,400 | ok |
| 7 comparisons | setups covered | 8 | 8 | ok |
| 8 stochasticity | SMER: rows | >0 | 40 | ok |
| 8 stochasticity | SMER: lengths measured | >1 | 2 | ok |
| 8 stochasticity | LIME: rows | >0 | 60 | ok |
| 8 stochasticity | LIME: lengths measured | >1 | 2 | ok |
| 8 stochasticity | caption stability: lengths | 8 | 8 | ok |
| 9 figures | AOPC png | 205 | 205 | ok |
| 9 figures | AOPC folders | 6 | 6 | ok |
| 9 figures | caption-length png | >0 | 1 | ok |
| 10 reports | markdown reports | >0 | 9 | ok |

Regenerate with `python scripts/16_audit.py`.

