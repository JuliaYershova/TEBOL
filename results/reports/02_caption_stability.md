# Caption generation stability

The captioner was asked five times per image at each length, temperature 1.0. This measures how much those five differ, from the raw caption files alone -- no classifier or explainer involved.

| requested words | mean actual | length sd | token Jaccard | content Jaccard | identical | class agreement |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 3.7 | 0.56 | 0.621 | 0.630 | 0.328 | 0.838 |
| 5 | 5.7 | 0.54 | 0.640 | 0.639 | 0.246 | 0.909 |
| 7 | 7.2 | 0.71 | 0.615 | 0.607 | 0.157 | 0.909 |
| 10 | 10.3 | 1.02 | 0.584 | 0.572 | 0.070 | 0.907 |
| 15 | 16.8 | 1.68 | 0.546 | 0.518 | 0.017 | 0.901 |
| 20 | 22.7 | 2.24 | 0.518 | 0.480 | 0.005 | 0.901 |
| 25 | 27.3 | 2.87 | 0.478 | 0.441 | 0.001 | 0.901 |
| 30 | 34.3 | 3.02 | 0.462 | 0.422 | 0.000 | 0.900 |

## What each column measures

Five repetitions per image give C(5,2) = 10 pairs; every pairwise score below is the mean over those 10, then averaged over images.

| column | how it is computed | range | reading |
|---|---|---|---|
| mean actual | token count of each caption, averaged over all captions at that length | -- | how far the model overshoots the requested word count |
| length sd | standard deviation of token count across one image's 5 repetitions, averaged over images | 0+ | 0 = the same length every time; larger = the model varies how much it writes |
| token Jaccard | \|A n B\| / \|A u B\| on the token sets of two repetitions | 0-1 | 1 = the five captions used exactly the same words; 0.5 = they share half |
| content Jaccard | the same after removing function words (`the`, `with`, `on`, ...) | 0-1 | isolates *what was described* from *how the sentence was built*; below the token score means the agreement was partly scaffolding |
| identical | share of the 10 pairs whose normalised strings match exactly | 0-1 | 1 = generation is effectively deterministic; 0 = no two runs ever coincide |
| class agreement | share of images where all 5 repetitions agree on whether any class term appears | 0-1 | 1 = the class is named every time or never; below 1 = for those images, whether the caption leaks its own label is a coin flip |

