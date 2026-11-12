# Worked SMER examples -- arm `caption`, 5-word captions

One caption per class. `z_j = beta . e(word_j)`; `b_0` is the fitted intercept; `logit lambda(w_j) = z_j + b_0` is what the model outputs for that word alone. The final row is the averaged caption vector, `SAVe`, and its `z` is the mean of the column above it -- so `mean(z_j) + b_0` is the caption logit exactly.

The caption is image n02206856_1815, repetition 3. Coefficients come from the cross-validation fold that held the caption out.

## ant_bee -- true class `bee`

> bee pollinating vibrant purple flowers

`n02206856_1815`, rep 3, held out in fold 1. Predicted **bee** at P = 0.9937.

| Word | Embedding | z_j | b_0 | Logit lambda(w_j) | P(ant) | P(bee) | LIME importance |
|---|---|---:|---:|---:|---:|---:|---:|
| bee | [-0.0001, -0.0003, ..., -0.0033] | 13.4518 | 0.2429 | 13.6946 | 0.0000 | 1.0000 | 0.1224 |
| pollinating | [-0.0001, 0.0115, ..., 0.0002] | 6.9856 | 0.2429 | 7.2285 | 0.0007 | 0.9993 | 0.0854 |
| vibrant | [-0.0000, 0.0040, ..., -0.0138] | -1.1084 | 0.2429 | -0.8655 | 0.7038 | 0.2962 | -0.0124 |
| purple | [0.0001, -0.0147, ..., 0.0308] | -0.5149 | 0.2429 | -0.2721 | 0.5676 | 0.4324 | -0.0083 |
| flowers | [0.0001, -0.0267, ..., -0.0028] | 5.2931 | 0.2429 | 5.5360 | 0.0039 | 0.9961 | 0.0689 |
| **SAVe(caption)** | [0.0000, -0.0053, ..., 0.0022] | 4.8214 | 0.2429 | 5.0643 | 0.0063 | 0.9937 | N/A |
