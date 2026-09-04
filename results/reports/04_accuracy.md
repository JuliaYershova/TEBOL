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
|  | 15 | 0.985 <sub>0.984-0.987</sub> | 0.850 <sub>0.843-0.857</sub> | **+0.135** |
|  | 20 | 0.980 <sub>0.978-0.982</sub> | 0.848 <sub>0.841-0.855</sub> | **+0.132** |
|  | 25 | 0.975 <sub>0.972-0.979</sub> | 0.849 <sub>0.843-0.854</sub> | **+0.127** |
|  | 30 | 0.972 <sub>0.970-0.974</sub> | 0.858 <sub>0.853-0.864</sub> | **+0.114** |
| ambulance_firetruck | 3 | 0.985 <sub>0.983-0.986</sub> | 0.865 <sub>0.858-0.871</sub> | **+0.120** |
|  | 5 | 0.985 <sub>0.983-0.987</sub> | 0.879 <sub>0.872-0.886</sub> | **+0.106** |
|  | 7 | 0.983 <sub>0.981-0.986</sub> | 0.876 <sub>0.870-0.881</sub> | **+0.107** |
|  | 10 | 0.982 <sub>0.981-0.984</sub> | 0.882 <sub>0.877-0.887</sub> | **+0.100** |
|  | 15 | 0.978 <sub>0.976-0.980</sub> | 0.886 <sub>0.881-0.891</sub> | **+0.092** |
|  | 20 | 0.973 <sub>0.971-0.976</sub> | 0.891 <sub>0.886-0.897</sub> | **+0.082** |
|  | 25 | 0.973 <sub>0.970-0.976</sub> | 0.895 <sub>0.889-0.901</sub> | **+0.078** |
|  | 30 | 0.973 <sub>0.970-0.976</sub> | 0.893 <sub>0.887-0.898</sub> | **+0.080** |
| ant_bee | 3 | 0.984 <sub>0.982-0.986</sub> | 0.884 <sub>0.878-0.889</sub> | **+0.100** |
|  | 5 | 0.985 <sub>0.983-0.987</sub> | 0.887 <sub>0.882-0.891</sub> | **+0.098** |
|  | 7 | 0.984 <sub>0.982-0.987</sub> | 0.903 <sub>0.898-0.908</sub> | **+0.082** |
|  | 10 | 0.977 <sub>0.975-0.980</sub> | 0.915 <sub>0.911-0.920</sub> | **+0.062** |
|  | 15 | 0.963 <sub>0.960-0.966</sub> | 0.920 <sub>0.915-0.924</sub> | **+0.043** |
|  | 20 | 0.950 <sub>0.946-0.953</sub> | 0.910 <sub>0.906-0.915</sub> | **+0.039** |
|  | 25 | 0.945 <sub>0.941-0.948</sub> | 0.908 <sub>0.904-0.913</sub> | **+0.037** |
|  | 30 | 0.937 <sub>0.933-0.941</sub> | 0.900 <sub>0.895-0.904</sub> | **+0.037** |
| cucumber_zucchini | 3 | 0.929 <sub>0.925-0.932</sub> | 0.707 <sub>0.701-0.713</sub> | **+0.222** |
|  | 5 | 0.941 <sub>0.937-0.945</sub> | 0.709 <sub>0.699-0.718</sub> | **+0.232** |
|  | 7 | 0.937 <sub>0.933-0.941</sub> | 0.692 <sub>0.684-0.700</sub> | **+0.245** |
|  | 10 | 0.933 <sub>0.930-0.936</sub> | 0.697 <sub>0.687-0.706</sub> | **+0.236** |
|  | 15 | 0.918 <sub>0.912-0.923</sub> | 0.706 <sub>0.696-0.716</sub> | **+0.212** |
|  | 20 | 0.897 <sub>0.892-0.902</sub> | 0.701 <sub>0.691-0.710</sub> | **+0.196** |
|  | 25 | 0.884 <sub>0.877-0.890</sub> | 0.702 <sub>0.693-0.712</sub> | **+0.181** |
|  | 30 | 0.858 <sub>0.849-0.866</sub> | 0.688 <sub>0.677-0.699</sub> | **+0.170** |
| hotpot_vase | 3 | 0.997 <sub>0.997-0.998</sub> | 0.996 <sub>0.995-0.997</sub> | +0.001 |
|  | 5 | 0.998 <sub>0.997-0.998</sub> | 0.997 <sub>0.996-0.998</sub> | +0.001 |
|  | 7 | 0.997 <sub>0.996-0.998</sub> | 0.996 <sub>0.995-0.998</sub> | +0.001 |
|  | 10 | 0.996 <sub>0.995-0.997</sub> | 0.994 <sub>0.993-0.996</sub> | +0.002 |
|  | 15 | 0.996 <sub>0.995-0.997</sub> | 0.995 <sub>0.994-0.997</sub> | +0.000 |
|  | 20 | 0.996 <sub>0.995-0.998</sub> | 0.995 <sub>0.993-0.996</sub> | +0.001 |
|  | 25 | 0.995 <sub>0.993-0.996</sub> | 0.995 <sub>0.993-0.996</sub> | +0.000 |
|  | 30 | 0.996 <sub>0.995-0.998</sub> | 0.995 <sub>0.994-0.997</sub> | +0.001 |

## Tag subset -- images ImageNet-Captions covers

Descriptions against human tags. All three columns run on the same images, so the comparison is not confounded by sample. Tags have no length: they are what the uploader wrote, and the column repeats down the block.

| pair | words | caption | without synonyms | tags | cap - tags | nosyn - tags |
|---|---:|---|---|---|---:|---:|
| acousticguitar_violin | 3 | 0.982 <sub>0.978-0.985</sub> | 0.784 <sub>0.774-0.794</sub> | 0.977 <sub>0.973-0.981</sub> | +0.005 | **-0.193** |
|  | 5 | 0.986 <sub>0.982-0.989</sub> | 0.756 <sub>0.745-0.767</sub> |  | +0.009 | **-0.221** |
|  | 7 | 0.985 <sub>0.981-0.989</sub> | 0.747 <sub>0.735-0.760</sub> |  | +0.008 | **-0.230** |
|  | 10 | 0.982 <sub>0.978-0.985</sub> | 0.756 <sub>0.742-0.770</sub> |  | +0.005 | **-0.221** |
|  | 15 | 0.971 <sub>0.966-0.976</sub> | 0.755 <sub>0.743-0.767</sub> |  | -0.006 | **-0.222** |
|  | 20 | 0.969 <sub>0.964-0.974</sub> | 0.759 <sub>0.747-0.770</sub> |  | -0.008 | **-0.218** |
|  | 25 | 0.958 <sub>0.952-0.964</sub> | 0.770 <sub>0.760-0.780</sub> |  | -0.019 | **-0.207** |
|  | 30 | 0.948 <sub>0.942-0.954</sub> | 0.788 <sub>0.778-0.798</sub> |  | -0.029 | **-0.189** |
| ambulance_firetruck | 3 | 0.983 <sub>0.980-0.987</sub> | 0.813 <sub>0.803-0.822</sub> | 0.993 <sub>0.990-0.996</sub> | -0.010 | **-0.180** |
|  | 5 | 0.992 <sub>0.990-0.995</sub> | 0.819 <sub>0.806-0.831</sub> |  | -0.001 | **-0.174** |
|  | 7 | 0.990 <sub>0.987-0.994</sub> | 0.831 <sub>0.821-0.842</sub> |  | -0.003 | **-0.162** |
|  | 10 | 0.989 <sub>0.985-0.992</sub> | 0.857 <sub>0.848-0.866</sub> |  | -0.004 | **-0.136** |
|  | 15 | 0.990 <sub>0.987-0.992</sub> | 0.879 <sub>0.869-0.889</sub> |  | -0.003 | **-0.114** |
|  | 20 | 0.983 <sub>0.980-0.986</sub> | 0.865 <sub>0.855-0.876</sub> |  | -0.010 | **-0.128** |
|  | 25 | 0.982 <sub>0.977-0.986</sub> | 0.870 <sub>0.859-0.882</sub> |  | -0.011 | **-0.123** |
|  | 30 | 0.979 <sub>0.974-0.984</sub> | 0.865 <sub>0.856-0.875</sub> |  | -0.014 | **-0.128** |
| ant_bee | 3 | 0.984 <sub>0.980-0.988</sub> | 0.866 <sub>0.860-0.873</sub> | 0.958 <sub>0.953-0.963</sub> | +0.026 | **-0.091** |
|  | 5 | 0.985 <sub>0.982-0.988</sub> | 0.862 <sub>0.855-0.868</sub> |  | +0.027 | **-0.096** |
|  | 7 | 0.980 <sub>0.976-0.983</sub> | 0.874 <sub>0.867-0.881</sub> |  | +0.022 | **-0.083** |
|  | 10 | 0.965 <sub>0.961-0.969</sub> | 0.891 <sub>0.885-0.897</sub> |  | +0.007 | **-0.067** |
|  | 15 | 0.944 <sub>0.939-0.950</sub> | 0.895 <sub>0.889-0.902</sub> |  | -0.013 | **-0.062** |
|  | 20 | 0.929 <sub>0.924-0.935</sub> | 0.892 <sub>0.885-0.898</sub> |  | -0.028 | **-0.066** |
|  | 25 | 0.921 <sub>0.915-0.927</sub> | 0.893 <sub>0.887-0.898</sub> |  | -0.036 | **-0.065** |
|  | 30 | 0.914 <sub>0.907-0.920</sub> | 0.884 <sub>0.877-0.891</sub> |  | **-0.044** | **-0.074** |
| cucumber_zucchini | 3 | 0.928 <sub>0.921-0.936</sub> | 0.707 <sub>0.697-0.718</sub> | 0.924 <sub>0.917-0.930</sub> | +0.004 | **-0.217** |
|  | 5 | 0.928 <sub>0.920-0.936</sub> | 0.698 <sub>0.689-0.707</sub> |  | +0.004 | **-0.226** |
|  | 7 | 0.926 <sub>0.919-0.934</sub> | 0.697 <sub>0.686-0.708</sub> |  | +0.002 | **-0.227** |
|  | 10 | 0.914 <sub>0.905-0.922</sub> | 0.695 <sub>0.683-0.707</sub> |  | -0.010 | **-0.229** |
|  | 15 | 0.889 <sub>0.880-0.899</sub> | 0.675 <sub>0.661-0.690</sub> |  | -0.035 | **-0.249** |
|  | 20 | 0.858 <sub>0.848-0.868</sub> | 0.668 <sub>0.656-0.680</sub> |  | **-0.066** | **-0.256** |
|  | 25 | 0.821 <sub>0.810-0.832</sub> | 0.665 <sub>0.653-0.677</sub> |  | **-0.103** | **-0.259** |
|  | 30 | 0.783 <sub>0.767-0.798</sub> | 0.655 <sub>0.644-0.667</sub> |  | **-0.141** | **-0.269** |
| hotpot_vase | 3 | 0.998 <sub>0.996-1.000</sub> | 0.996 <sub>0.994-0.998</sub> | 0.971 <sub>0.966-0.976</sub> | +0.027 | +0.025 |
|  | 5 | 0.998 <sub>0.996-1.000</sub> | 0.998 <sub>0.996-1.000</sub> |  | +0.027 | +0.027 |
|  | 7 | 0.998 <sub>0.996-1.000</sub> | 0.998 <sub>0.996-1.000</sub> |  | +0.027 | +0.027 |
|  | 10 | 0.996 <sub>0.993-0.998</sub> | 0.996 <sub>0.993-0.998</sub> |  | +0.025 | +0.025 |
|  | 15 | 0.998 <sub>0.996-1.000</sub> | 0.998 <sub>0.996-1.000</sub> |  | +0.027 | +0.027 |
|  | 20 | 0.996 <sub>0.993-0.998</sub> | 0.996 <sub>0.993-0.998</sub> |  | +0.025 | +0.025 |
|  | 25 | 0.998 <sub>0.996-1.000</sub> | 0.998 <sub>0.996-1.000</sub> |  | +0.027 | +0.027 |
|  | 30 | 0.998 <sub>0.996-1.000</sub> | 0.998 <sub>0.996-1.000</sub> |  | +0.027 | +0.027 |

## Reading these

- Tags are never synonym-stripped -- a tag set is what a human wrote about the photograph, and ablating it would answer a question nobody asked. So `cap - tags` compares a leaky caption to a leaky tag set, and `nosyn - tags` compares a clean caption to a leaky tag set. Neither is a like-for-like contest; both are worth reporting.
- The synonym removal strips only a caption's *own* class terms, so cross-class mentions survive and become inverted indicators. See `class_leakage.md`.
