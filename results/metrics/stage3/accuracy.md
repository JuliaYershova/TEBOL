# Stage 3 -- accuracy, image level

Mean over 25 fits (5-fold x 5 repeats), with the across-fold 95% interval. Folds are assigned to *images*, so the five captions of one photograph never straddle the split, and every arm reuses the same assignment -- which is what makes the deltas paired.

`delta` columns are paired differences; **bold** is significant under the Nadeau-Bengio corrected resampled t-test with Bonferroni correction inside the comparison family. The correction is applied across all pairs and lengths in a family, so it is conservative.

## Full scope -- every captioned image

The synonym ablation. Both columns run on all images of the pair, so the delta is the cost of removing the class name and nothing else.

| pair | words | caption | without synonyms | delta |
|---|---:|---|---|---:|
| acousticguitar_violin | 3 | 0.986 <sub>0.984-0.988</sub> | 0.823 <sub>0.816-0.830</sub> | **+0.163** |
|  | 5 | 0.991 <sub>0.989-0.992</sub> | 0.818 <sub>0.812-0.824</sub> | **+0.173** |
|  | 7 | 0.990 <sub>0.989-0.992</sub> | 0.818 <sub>0.812-0.825</sub> | **+0.172** |
|  | 10 | 0.989 <sub>0.987-0.991</sub> | 0.838 <sub>0.832-0.844</sub> | **+0.151** |
| ambulance_firetruck | 3 | 0.985 <sub>0.983-0.986</sub> | 0.865 <sub>0.858-0.871</sub> | **+0.120** |
|  | 5 | 0.985 <sub>0.983-0.987</sub> | 0.879 <sub>0.872-0.886</sub> | **+0.106** |
|  | 7 | 0.983 <sub>0.981-0.986</sub> | 0.876 <sub>0.870-0.881</sub> | **+0.107** |
|  | 10 | 0.982 <sub>0.981-0.984</sub> | 0.882 <sub>0.877-0.887</sub> | **+0.100** |
| ant_bee | 3 | 0.984 <sub>0.982-0.986</sub> | 0.884 <sub>0.878-0.889</sub> | **+0.100** |
|  | 5 | 0.985 <sub>0.983-0.987</sub> | 0.887 <sub>0.882-0.891</sub> | **+0.098** |
|  | 7 | 0.984 <sub>0.982-0.987</sub> | 0.903 <sub>0.898-0.908</sub> | **+0.082** |
|  | 10 | 0.977 <sub>0.975-0.980</sub> | 0.915 <sub>0.911-0.920</sub> | **+0.062** |
| cucumber_zucchini | 3 | 0.929 <sub>0.925-0.932</sub> | 0.707 <sub>0.701-0.713</sub> | **+0.222** |
|  | 5 | 0.941 <sub>0.937-0.945</sub> | 0.709 <sub>0.699-0.718</sub> | **+0.232** |
|  | 7 | 0.937 <sub>0.933-0.941</sub> | 0.692 <sub>0.684-0.700</sub> | **+0.245** |
|  | 10 | 0.933 <sub>0.930-0.936</sub> | 0.697 <sub>0.687-0.706</sub> | **+0.236** |
| hotpot_vase | 3 | 0.997 <sub>0.997-0.998</sub> | 0.996 <sub>0.995-0.997</sub> | +0.001 |
|  | 5 | 0.998 <sub>0.997-0.998</sub> | 0.997 <sub>0.996-0.998</sub> | +0.001 |
|  | 7 | 0.997 <sub>0.996-0.998</sub> | 0.996 <sub>0.995-0.998</sub> | +0.001 |
|  | 10 | 0.996 <sub>0.995-0.997</sub> | 0.994 <sub>0.993-0.996</sub> | +0.002 |

## Tag subset -- images ImageNet-Captions covers

Descriptions against human tags. All three columns run on the same images, so the comparison is not confounded by sample. Tags have no length: they are what the uploader wrote, and the column repeats down the block.

| pair | words | caption | without synonyms | tags | cap - tags | nosyn - tags |
|---|---:|---|---|---|---:|---:|
| acousticguitar_violin | 3 | 0.982 <sub>0.978-0.985</sub> | 0.784 <sub>0.774-0.794</sub> | 0.977 <sub>0.973-0.981</sub> | +0.005 | **-0.193** |
|  | 5 | 0.986 <sub>0.982-0.989</sub> | 0.756 <sub>0.745-0.767</sub> |  | +0.009 | **-0.221** |
|  | 7 | 0.985 <sub>0.981-0.989</sub> | 0.747 <sub>0.735-0.760</sub> |  | +0.008 | **-0.230** |
|  | 10 | 0.982 <sub>0.978-0.985</sub> | 0.756 <sub>0.742-0.770</sub> |  | +0.005 | **-0.221** |
| ambulance_firetruck | 3 | 0.983 <sub>0.980-0.987</sub> | 0.813 <sub>0.803-0.822</sub> | 0.993 <sub>0.990-0.996</sub> | -0.010 | **-0.180** |
|  | 5 | 0.992 <sub>0.990-0.995</sub> | 0.819 <sub>0.806-0.831</sub> |  | -0.001 | **-0.174** |
|  | 7 | 0.990 <sub>0.987-0.994</sub> | 0.831 <sub>0.821-0.842</sub> |  | -0.003 | **-0.162** |
|  | 10 | 0.989 <sub>0.985-0.992</sub> | 0.857 <sub>0.848-0.866</sub> |  | -0.004 | **-0.136** |
| ant_bee | 3 | 0.984 <sub>0.980-0.988</sub> | 0.866 <sub>0.860-0.873</sub> | 0.958 <sub>0.953-0.963</sub> | +0.026 | **-0.091** |
|  | 5 | 0.985 <sub>0.982-0.988</sub> | 0.862 <sub>0.855-0.868</sub> |  | **+0.027** | **-0.096** |
|  | 7 | 0.980 <sub>0.976-0.983</sub> | 0.874 <sub>0.867-0.881</sub> |  | +0.022 | **-0.083** |
|  | 10 | 0.965 <sub>0.961-0.969</sub> | 0.891 <sub>0.885-0.897</sub> |  | +0.007 | **-0.067** |
| cucumber_zucchini | 3 | 0.928 <sub>0.921-0.936</sub> | 0.707 <sub>0.697-0.718</sub> | 0.924 <sub>0.917-0.930</sub> | +0.004 | **-0.217** |
|  | 5 | 0.928 <sub>0.920-0.936</sub> | 0.698 <sub>0.689-0.707</sub> |  | +0.004 | **-0.226** |
|  | 7 | 0.926 <sub>0.919-0.934</sub> | 0.697 <sub>0.686-0.708</sub> |  | +0.002 | **-0.227** |
|  | 10 | 0.914 <sub>0.905-0.922</sub> | 0.695 <sub>0.683-0.707</sub> |  | -0.010 | **-0.229** |
| hotpot_vase | 3 | 0.998 <sub>0.996-1.000</sub> | 0.996 <sub>0.994-0.998</sub> | 0.971 <sub>0.966-0.976</sub> | **+0.027** | **+0.025** |
|  | 5 | 0.998 <sub>0.996-1.000</sub> | 0.998 <sub>0.996-1.000</sub> |  | **+0.027** | **+0.027** |
|  | 7 | 0.998 <sub>0.996-1.000</sub> | 0.998 <sub>0.996-1.000</sub> |  | **+0.027** | **+0.027** |
|  | 10 | 0.996 <sub>0.993-0.998</sub> | 0.996 <sub>0.993-0.998</sub> |  | **+0.025** | **+0.025** |

## Reading these

- Tags are never synonym-stripped -- a tag set is what a human wrote about the photograph, and ablating it would answer a question nobody asked. So `cap - tags` compares a leaky caption to a leaky tag set, and `nosyn - tags` compares a clean caption to a leaky tag set. Neither is a like-for-like contest; both are worth reporting.
- The synonym removal strips only a caption's *own* class terms, so cross-class mentions survive and become inverted indicators. See `class_leakage.md`.
