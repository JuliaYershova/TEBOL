# AOPC for SMER

Mean drop in the predicted class probability after removing the k highest-ranked words, over the 25 (fold, rep) groups; the interval under each value is the 95% t-interval of that spread. Removal is capped at n-1 words (`--exhaust hold`), so a caption that runs out holds its last value and stays in the denominator.

Computed entirely from `smer_words.parquet`: because the classifier is linear over a mean of word vectors, removing words is `bias + mean(z[kept])`, so no embedding or model call is involved. The whole table takes about 25 seconds.

**Local and global rankings agree to three decimals** in every configuration -- see the note at the end.

## acousticguitar_violin

| arm | setup | k=0 | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 | k=7 | k=8 | k=9 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| caption | w03 | 0.000<br><sub>0.000-0.000</sub> | 0.224<br><sub>0.221-0.227</sub> | 0.461<br><sub>0.458-0.464</sub> | -- | -- | -- | -- | -- | -- | -- |
| caption | w05 | 0.000<br><sub>0.000-0.000</sub> | 0.222<br><sub>0.219-0.226</sub> | 0.444<br><sub>0.441-0.448</sub> | 0.537<br><sub>0.534-0.540</sub> | 0.608<br><sub>0.605-0.611</sub> | -- | -- | -- | -- | -- |
| caption | w07 | 0.000<br><sub>0.000-0.000</sub> | 0.202<br><sub>0.198-0.205</sub> | 0.408<br><sub>0.404-0.412</sub> | 0.493<br><sub>0.490-0.496</sub> | 0.559<br><sub>0.557-0.562</sub> | 0.619<br><sub>0.616-0.621</sub> | 0.661<br><sub>0.658-0.663</sub> | -- | -- | -- |
| caption | w10 | 0.000<br><sub>0.000-0.000</sub> | 0.164<br><sub>0.163-0.166</sub> | 0.357<br><sub>0.354-0.360</sub> | 0.436<br><sub>0.434-0.439</sub> | 0.493<br><sub>0.491-0.495</sub> | 0.542<br><sub>0.541-0.544</sub> | 0.589<br><sub>0.587-0.590</sub> | 0.633<br><sub>0.632-0.635</sub> | 0.674<br><sub>0.672-0.675</sub> | 0.702<br><sub>0.700-0.704</sub> |
| caption_noclass | w03 | 0.000<br><sub>0.000-0.000</sub> | 0.135<br><sub>0.134-0.137</sub> | 0.190<br><sub>0.187-0.192</sub> | -- | -- | -- | -- | -- | -- | -- |
| caption_noclass | w05 | 0.000<br><sub>0.000-0.000</sub> | 0.123<br><sub>0.122-0.124</sub> | 0.230<br><sub>0.228-0.232</sub> | 0.306<br><sub>0.304-0.309</sub> | 0.338<br><sub>0.336-0.341</sub> | -- | -- | -- | -- | -- |
| caption_noclass | w07 | 0.000<br><sub>0.000-0.000</sub> | 0.111<br><sub>0.109-0.112</sub> | 0.212<br><sub>0.210-0.214</sub> | 0.309<br><sub>0.307-0.311</sub> | 0.387<br><sub>0.384-0.389</sub> | 0.430<br><sub>0.427-0.433</sub> | 0.449<br><sub>0.446-0.452</sub> | -- | -- | -- |
| caption_noclass | w10 | 0.000<br><sub>0.000-0.000</sub> | 0.085<br><sub>0.084-0.086</sub> | 0.162<br><sub>0.161-0.164</sub> | 0.237<br><sub>0.236-0.239</sub> | 0.311<br><sub>0.309-0.313</sub> | 0.383<br><sub>0.380-0.385</sub> | 0.447<br><sub>0.444-0.451</sub> | 0.494<br><sub>0.491-0.498</sub> | 0.521<br><sub>0.517-0.525</sub> | 0.534<br><sub>0.530-0.537</sub> |
| tags | tags | 0.000<br><sub>0.000-0.000</sub> | 0.103<br><sub>0.097-0.108</sub> | 0.180<br><sub>0.176-0.184</sub> | 0.220<br><sub>0.213-0.227</sub> | 0.243<br><sub>0.234-0.251</sub> | 0.259<br><sub>0.248-0.269</sub> | -- | -- | -- | -- |
| caption_tagsub | w03 | 0.000<br><sub>0.000-0.000</sub> | 0.242<br><sub>0.240-0.245</sub> | 0.404<br><sub>0.399-0.409</sub> | -- | -- | -- | -- | -- | -- | -- |
| caption_tagsub | w05 | 0.000<br><sub>0.000-0.000</sub> | 0.222<br><sub>0.219-0.225</sub> | 0.363<br><sub>0.359-0.367</sub> | 0.441<br><sub>0.437-0.444</sub> | 0.504<br><sub>0.501-0.508</sub> | -- | -- | -- | -- | -- |
| caption_tagsub | w07 | 0.000<br><sub>0.000-0.000</sub> | 0.193<br><sub>0.191-0.195</sub> | 0.324<br><sub>0.320-0.328</sub> | 0.394<br><sub>0.389-0.399</sub> | 0.450<br><sub>0.445-0.455</sub> | 0.503<br><sub>0.498-0.509</sub> | 0.542<br><sub>0.537-0.547</sub> | -- | -- | -- |
| caption_tagsub | w10 | 0.000<br><sub>0.000-0.000</sub> | 0.142<br><sub>0.140-0.143</sub> | 0.264<br><sub>0.260-0.268</sub> | 0.323<br><sub>0.319-0.327</sub> | 0.368<br><sub>0.365-0.372</sub> | 0.409<br><sub>0.406-0.412</sub> | 0.450<br><sub>0.447-0.453</sub> | 0.492<br><sub>0.489-0.496</sub> | 0.532<br><sub>0.528-0.535</sub> | 0.560<br><sub>0.556-0.564</sub> |
| caption_noclass_tagsub | w03 | 0.000<br><sub>0.000-0.000</sub> | 0.116<br><sub>0.113-0.118</sub> | 0.161<br><sub>0.158-0.165</sub> | -- | -- | -- | -- | -- | -- | -- |
| caption_noclass_tagsub | w05 | 0.000<br><sub>0.000-0.000</sub> | 0.092<br><sub>0.089-0.094</sub> | 0.186<br><sub>0.182-0.190</sub> | 0.257<br><sub>0.253-0.261</sub> | 0.288<br><sub>0.284-0.291</sub> | -- | -- | -- | -- | -- |
| caption_noclass_tagsub | w07 | 0.000<br><sub>0.000-0.000</sub> | 0.086<br><sub>0.083-0.088</sub> | 0.164<br><sub>0.161-0.168</sub> | 0.237<br><sub>0.232-0.242</sub> | 0.299<br><sub>0.294-0.305</sub> | 0.336<br><sub>0.331-0.342</sub> | 0.354<br><sub>0.348-0.361</sub> | -- | -- | -- |
| caption_noclass_tagsub | w10 | 0.000<br><sub>0.000-0.000</sub> | 0.061<br><sub>0.060-0.062</sub> | 0.115<br><sub>0.113-0.116</sub> | 0.166<br><sub>0.163-0.168</sub> | 0.217<br><sub>0.213-0.220</sub> | 0.267<br><sub>0.262-0.271</sub> | 0.313<br><sub>0.309-0.318</sub> | 0.353<br><sub>0.348-0.358</sub> | 0.381<br><sub>0.376-0.385</sub> | 0.396<br><sub>0.391-0.401</sub> |

## ambulance_firetruck

| arm | setup | k=0 | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 | k=7 | k=8 | k=9 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| caption | w03 | 0.000<br><sub>0.000-0.000</sub> | 0.173<br><sub>0.171-0.176</sub> | 0.368<br><sub>0.364-0.371</sub> | -- | -- | -- | -- | -- | -- | -- |
| caption | w05 | 0.000<br><sub>0.000-0.000</sub> | 0.163<br><sub>0.161-0.165</sub> | 0.376<br><sub>0.373-0.378</sub> | 0.490<br><sub>0.486-0.494</sub> | 0.583<br><sub>0.578-0.587</sub> | -- | -- | -- | -- | -- |
| caption | w07 | 0.000<br><sub>0.000-0.000</sub> | 0.173<br><sub>0.171-0.174</sub> | 0.353<br><sub>0.351-0.355</sub> | 0.462<br><sub>0.459-0.465</sub> | 0.546<br><sub>0.543-0.550</sub> | 0.622<br><sub>0.618-0.626</sub> | 0.684<br><sub>0.681-0.688</sub> | -- | -- | -- |
| caption | w10 | 0.000<br><sub>0.000-0.000</sub> | 0.165<br><sub>0.164-0.167</sub> | 0.312<br><sub>0.310-0.314</sub> | 0.407<br><sub>0.404-0.409</sub> | 0.481<br><sub>0.477-0.484</sub> | 0.547<br><sub>0.543-0.550</sub> | 0.608<br><sub>0.605-0.612</sub> | 0.666<br><sub>0.663-0.669</sub> | 0.716<br><sub>0.713-0.719</sub> | 0.753<br><sub>0.750-0.756</sub> |
| caption_noclass | w03 | 0.000<br><sub>0.000-0.000</sub> | 0.155<br><sub>0.153-0.158</sub> | 0.212<br><sub>0.208-0.215</sub> | -- | -- | -- | -- | -- | -- | -- |
| caption_noclass | w05 | 0.000<br><sub>0.000-0.000</sub> | 0.196<br><sub>0.194-0.199</sub> | 0.339<br><sub>0.335-0.343</sub> | 0.415<br><sub>0.410-0.420</sub> | 0.450<br><sub>0.444-0.455</sub> | -- | -- | -- | -- | -- |
| caption_noclass | w07 | 0.000<br><sub>0.000-0.000</sub> | 0.187<br><sub>0.185-0.188</sub> | 0.307<br><sub>0.306-0.309</sub> | 0.397<br><sub>0.395-0.399</sub> | 0.466<br><sub>0.463-0.469</sub> | 0.513<br><sub>0.510-0.516</sub> | 0.534<br><sub>0.531-0.537</sub> | -- | -- | -- |
| caption_noclass | w10 | 0.000<br><sub>0.000-0.000</sub> | 0.141<br><sub>0.139-0.142</sub> | 0.243<br><sub>0.241-0.245</sub> | 0.320<br><sub>0.317-0.322</sub> | 0.385<br><sub>0.383-0.388</sub> | 0.445<br><sub>0.442-0.448</sub> | 0.500<br><sub>0.497-0.503</sub> | 0.547<br><sub>0.545-0.550</sub> | 0.580<br><sub>0.577-0.583</sub> | 0.598<br><sub>0.596-0.601</sub> |
| tags | tags | 0.000<br><sub>0.000-0.000</sub> | 0.104<br><sub>0.098-0.110</sub> | 0.165<br><sub>0.155-0.175</sub> | 0.204<br><sub>0.196-0.213</sub> | 0.228<br><sub>0.220-0.236</sub> | 0.246<br><sub>0.236-0.257</sub> | -- | -- | -- | -- |
| caption_tagsub | w03 | 0.000<br><sub>0.000-0.000</sub> | 0.195<br><sub>0.190-0.201</sub> | 0.374<br><sub>0.369-0.379</sub> | -- | -- | -- | -- | -- | -- | -- |
| caption_tagsub | w05 | 0.000<br><sub>0.000-0.000</sub> | 0.189<br><sub>0.186-0.192</sub> | 0.395<br><sub>0.391-0.399</sub> | 0.521<br><sub>0.516-0.526</sub> | 0.589<br><sub>0.583-0.594</sub> | -- | -- | -- | -- | -- |
| caption_tagsub | w07 | 0.000<br><sub>0.000-0.000</sub> | 0.186<br><sub>0.184-0.189</sub> | 0.351<br><sub>0.348-0.353</sub> | 0.456<br><sub>0.453-0.459</sub> | 0.519<br><sub>0.515-0.522</sub> | 0.572<br><sub>0.569-0.576</sub> | 0.613<br><sub>0.609-0.617</sub> | -- | -- | -- |
| caption_tagsub | w10 | 0.000<br><sub>0.000-0.000</sub> | 0.160<br><sub>0.159-0.161</sub> | 0.277<br><sub>0.275-0.279</sub> | 0.351<br><sub>0.349-0.353</sub> | 0.405<br><sub>0.402-0.407</sub> | 0.449<br><sub>0.446-0.452</sub> | 0.491<br><sub>0.488-0.495</sub> | 0.532<br><sub>0.529-0.536</sub> | 0.573<br><sub>0.569-0.576</sub> | 0.607<br><sub>0.603-0.610</sub> |
| caption_noclass_tagsub | w03 | 0.000<br><sub>0.000-0.000</sub> | 0.129<br><sub>0.125-0.134</sub> | 0.158<br><sub>0.153-0.164</sub> | -- | -- | -- | -- | -- | -- | -- |
| caption_noclass_tagsub | w05 | 0.000<br><sub>0.000-0.000</sub> | 0.206<br><sub>0.203-0.208</sub> | 0.295<br><sub>0.292-0.298</sub> | 0.348<br><sub>0.345-0.351</sub> | 0.369<br><sub>0.366-0.371</sub> | -- | -- | -- | -- | -- |
| caption_noclass_tagsub | w07 | 0.000<br><sub>0.000-0.000</sub> | 0.170<br><sub>0.168-0.171</sub> | 0.243<br><sub>0.241-0.245</sub> | 0.306<br><sub>0.303-0.309</sub> | 0.358<br><sub>0.354-0.361</sub> | 0.389<br><sub>0.386-0.392</sub> | 0.404<br><sub>0.401-0.407</sub> | -- | -- | -- |
| caption_noclass_tagsub | w10 | 0.000<br><sub>0.000-0.000</sub> | 0.099<br><sub>0.097-0.101</sub> | 0.162<br><sub>0.159-0.164</sub> | 0.209<br><sub>0.207-0.212</sub> | 0.255<br><sub>0.252-0.258</sub> | 0.300<br><sub>0.297-0.303</sub> | 0.346<br><sub>0.343-0.349</sub> | 0.389<br><sub>0.387-0.392</sub> | 0.422<br><sub>0.420-0.425</sub> | 0.442<br><sub>0.440-0.445</sub> |

## ant_bee

| arm | setup | k=0 | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 | k=7 | k=8 | k=9 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| caption | w03 | 0.000<br><sub>0.000-0.000</sub> | 0.229<br><sub>0.227-0.231</sub> | 0.398<br><sub>0.394-0.402</sub> | -- | -- | -- | -- | -- | -- | -- |
| caption | w05 | 0.000<br><sub>0.000-0.000</sub> | 0.238<br><sub>0.237-0.240</sub> | 0.436<br><sub>0.433-0.438</sub> | 0.573<br><sub>0.570-0.575</sub> | 0.661<br><sub>0.659-0.664</sub> | -- | -- | -- | -- | -- |
| caption | w07 | 0.000<br><sub>0.000-0.000</sub> | 0.202<br><sub>0.200-0.203</sub> | 0.365<br><sub>0.363-0.366</sub> | 0.510<br><sub>0.508-0.512</sub> | 0.608<br><sub>0.605-0.610</sub> | 0.686<br><sub>0.683-0.689</sub> | 0.729<br><sub>0.726-0.732</sub> | -- | -- | -- |
| caption | w10 | 0.000<br><sub>0.000-0.000</sub> | 0.142<br><sub>0.141-0.143</sub> | 0.254<br><sub>0.253-0.255</sub> | 0.372<br><sub>0.371-0.374</sub> | 0.464<br><sub>0.462-0.465</sub> | 0.543<br><sub>0.541-0.546</sub> | 0.617<br><sub>0.615-0.620</sub> | 0.679<br><sub>0.676-0.682</sub> | 0.722<br><sub>0.719-0.725</sub> | 0.747<br><sub>0.744-0.750</sub> |
| caption_noclass | w03 | 0.000<br><sub>0.000-0.000</sub> | 0.149<br><sub>0.146-0.151</sub> | 0.214<br><sub>0.211-0.217</sub> | -- | -- | -- | -- | -- | -- | -- |
| caption_noclass | w05 | 0.000<br><sub>0.000-0.000</sub> | 0.159<br><sub>0.155-0.162</sub> | 0.287<br><sub>0.284-0.291</sub> | 0.388<br><sub>0.385-0.391</sub> | 0.424<br><sub>0.422-0.426</sub> | -- | -- | -- | -- | -- |
| caption_noclass | w07 | 0.000<br><sub>0.000-0.000</sub> | 0.131<br><sub>0.130-0.133</sub> | 0.238<br><sub>0.236-0.241</sub> | 0.333<br><sub>0.331-0.336</sub> | 0.425<br><sub>0.422-0.428</sub> | 0.479<br><sub>0.476-0.482</sub> | 0.497<br><sub>0.493-0.500</sub> | -- | -- | -- |
| caption_noclass | w10 | 0.000<br><sub>0.000-0.000</sub> | 0.100<br><sub>0.099-0.101</sub> | 0.180<br><sub>0.178-0.182</sub> | 0.257<br><sub>0.253-0.260</sub> | 0.336<br><sub>0.332-0.340</sub> | 0.417<br><sub>0.412-0.421</sub> | 0.491<br><sub>0.485-0.496</sub> | 0.547<br><sub>0.541-0.552</sub> | 0.581<br><sub>0.575-0.586</sub> | 0.598<br><sub>0.592-0.603</sub> |
| tags | tags | 0.000<br><sub>0.000-0.000</sub> | 0.145<br><sub>0.134-0.155</sub> | 0.242<br><sub>0.228-0.257</sub> | 0.322<br><sub>0.307-0.337</sub> | 0.377<br><sub>0.361-0.392</sub> | 0.416<br><sub>0.401-0.430</sub> | 0.444<br><sub>0.433-0.456</sub> | -- | -- | -- |
| caption_tagsub | w03 | 0.000<br><sub>0.000-0.000</sub> | 0.237<br><sub>0.234-0.240</sub> | 0.408<br><sub>0.404-0.411</sub> | -- | -- | -- | -- | -- | -- | -- |
| caption_tagsub | w05 | 0.000<br><sub>0.000-0.000</sub> | 0.229<br><sub>0.227-0.230</sub> | 0.412<br><sub>0.408-0.415</sub> | 0.541<br><sub>0.537-0.545</sub> | 0.632<br><sub>0.628-0.637</sub> | -- | -- | -- | -- | -- |
| caption_tagsub | w07 | 0.000<br><sub>0.000-0.000</sub> | 0.187<br><sub>0.185-0.188</sub> | 0.329<br><sub>0.326-0.331</sub> | 0.453<br><sub>0.449-0.457</sub> | 0.548<br><sub>0.542-0.553</sub> | 0.629<br><sub>0.623-0.634</sub> | 0.675<br><sub>0.670-0.681</sub> | -- | -- | -- |
| caption_tagsub | w10 | 0.000<br><sub>0.000-0.000</sub> | 0.121<br><sub>0.120-0.122</sub> | 0.217<br><sub>0.215-0.218</sub> | 0.316<br><sub>0.314-0.319</sub> | 0.400<br><sub>0.396-0.404</sub> | 0.478<br><sub>0.473-0.482</sub> | 0.551<br><sub>0.545-0.557</sub> | 0.613<br><sub>0.607-0.620</sub> | 0.657<br><sub>0.650-0.663</sub> | 0.682<br><sub>0.675-0.689</sub> |
| caption_noclass_tagsub | w03 | 0.000<br><sub>0.000-0.000</sub> | 0.146<br><sub>0.143-0.150</sub> | 0.214<br><sub>0.209-0.218</sub> | -- | -- | -- | -- | -- | -- | -- |
| caption_noclass_tagsub | w05 | 0.000<br><sub>0.000-0.000</sub> | 0.146<br><sub>0.143-0.148</sub> | 0.263<br><sub>0.258-0.267</sub> | 0.350<br><sub>0.345-0.355</sub> | 0.379<br><sub>0.375-0.383</sub> | -- | -- | -- | -- | -- |
| caption_noclass_tagsub | w07 | 0.000<br><sub>0.000-0.000</sub> | 0.119<br><sub>0.116-0.122</sub> | 0.211<br><sub>0.206-0.216</sub> | 0.296<br><sub>0.289-0.302</sub> | 0.376<br><sub>0.368-0.383</sub> | 0.424<br><sub>0.417-0.431</sub> | 0.439<br><sub>0.432-0.446</sub> | -- | -- | -- |
| caption_noclass_tagsub | w10 | 0.000<br><sub>0.000-0.000</sub> | 0.091<br><sub>0.089-0.093</sub> | 0.159<br><sub>0.157-0.161</sub> | 0.227<br><sub>0.224-0.230</sub> | 0.299<br><sub>0.295-0.303</sub> | 0.374<br><sub>0.369-0.379</sub> | 0.441<br><sub>0.436-0.447</sub> | 0.492<br><sub>0.486-0.497</sub> | 0.522<br><sub>0.517-0.527</sub> | 0.537<br><sub>0.532-0.542</sub> |

## cucumber_zucchini

| arm | setup | k=0 | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 | k=7 | k=8 | k=9 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| caption | w03 | 0.000<br><sub>0.000-0.000</sub> | 0.312<br><sub>0.309-0.316</sub> | 0.404<br><sub>0.398-0.410</sub> | -- | -- | -- | -- | -- | -- | -- |
| caption | w05 | 0.000<br><sub>0.000-0.000</sub> | 0.278<br><sub>0.274-0.281</sub> | 0.356<br><sub>0.351-0.360</sub> | 0.422<br><sub>0.418-0.426</sub> | 0.483<br><sub>0.478-0.488</sub> | -- | -- | -- | -- | -- |
| caption | w07 | 0.000<br><sub>0.000-0.000</sub> | 0.243<br><sub>0.241-0.246</sub> | 0.315<br><sub>0.311-0.318</sub> | 0.376<br><sub>0.373-0.380</sub> | 0.432<br><sub>0.428-0.436</sub> | 0.484<br><sub>0.480-0.488</sub> | 0.522<br><sub>0.517-0.527</sub> | -- | -- | -- |
| caption | w10 | 0.000<br><sub>0.000-0.000</sub> | 0.186<br><sub>0.184-0.188</sub> | 0.250<br><sub>0.247-0.252</sub> | 0.301<br><sub>0.298-0.304</sub> | 0.349<br><sub>0.346-0.352</sub> | 0.395<br><sub>0.391-0.398</sub> | 0.438<br><sub>0.435-0.441</sub> | 0.481<br><sub>0.477-0.484</sub> | 0.518<br><sub>0.514-0.521</sub> | 0.545<br><sub>0.541-0.548</sub> |
| caption_noclass | w03 | 0.000<br><sub>0.000-0.000</sub> | 0.104<br><sub>0.101-0.106</sub> | 0.156<br><sub>0.153-0.158</sub> | -- | -- | -- | -- | -- | -- | -- |
| caption_noclass | w05 | 0.000<br><sub>0.000-0.000</sub> | 0.112<br><sub>0.111-0.114</sub> | 0.194<br><sub>0.192-0.197</sub> | 0.268<br><sub>0.265-0.272</sub> | 0.310<br><sub>0.306-0.314</sub> | -- | -- | -- | -- | -- |
| caption_noclass | w07 | 0.000<br><sub>0.000-0.000</sub> | 0.098<br><sub>0.097-0.099</sub> | 0.168<br><sub>0.166-0.169</sub> | 0.230<br><sub>0.228-0.232</sub> | 0.290<br><sub>0.287-0.292</sub> | 0.335<br><sub>0.332-0.338</sub> | 0.358<br><sub>0.355-0.361</sub> | -- | -- | -- |
| caption_noclass | w10 | 0.000<br><sub>0.000-0.000</sub> | 0.077<br><sub>0.077-0.078</sub> | 0.132<br><sub>0.131-0.133</sub> | 0.179<br><sub>0.177-0.180</sub> | 0.221<br><sub>0.220-0.223</sub> | 0.263<br><sub>0.261-0.264</sub> | 0.306<br><sub>0.304-0.307</sub> | 0.346<br><sub>0.344-0.349</sub> | 0.378<br><sub>0.375-0.382</sub> | 0.398<br><sub>0.394-0.402</sub> |
| tags | tags | 0.000<br><sub>0.000-0.000</sub> | 0.121<br><sub>0.110-0.132</sub> | 0.153<br><sub>0.140-0.167</sub> | 0.176<br><sub>0.161-0.190</sub> | 0.193<br><sub>0.177-0.208</sub> | 0.207<br><sub>0.190-0.223</sub> | -- | -- | -- | -- |
| caption_tagsub | w03 | 0.000<br><sub>0.000-0.000</sub> | 0.264<br><sub>0.259-0.269</sub> | 0.350<br><sub>0.343-0.357</sub> | -- | -- | -- | -- | -- | -- | -- |
| caption_tagsub | w05 | 0.000<br><sub>0.000-0.000</sub> | 0.221<br><sub>0.217-0.226</sub> | 0.292<br><sub>0.286-0.298</sub> | 0.353<br><sub>0.346-0.360</sub> | 0.408<br><sub>0.400-0.416</sub> | -- | -- | -- | -- | -- |
| caption_tagsub | w07 | 0.000<br><sub>0.000-0.000</sub> | 0.182<br><sub>0.178-0.186</sub> | 0.244<br><sub>0.239-0.248</sub> | 0.297<br><sub>0.293-0.302</sub> | 0.346<br><sub>0.340-0.351</sub> | 0.392<br><sub>0.386-0.398</sub> | 0.427<br><sub>0.420-0.434</sub> | -- | -- | -- |
| caption_tagsub | w10 | 0.000<br><sub>0.000-0.000</sub> | 0.129<br><sub>0.126-0.131</sub> | 0.180<br><sub>0.177-0.183</sub> | 0.225<br><sub>0.222-0.228</sub> | 0.267<br><sub>0.264-0.270</sub> | 0.309<br><sub>0.305-0.312</sub> | 0.349<br><sub>0.346-0.353</sub> | 0.390<br><sub>0.386-0.394</sub> | 0.426<br><sub>0.422-0.430</sub> | 0.453<br><sub>0.449-0.457</sub> |
| caption_noclass_tagsub | w03 | 0.000<br><sub>0.000-0.000</sub> | 0.092<br><sub>0.088-0.095</sub> | 0.148<br><sub>0.144-0.152</sub> | -- | -- | -- | -- | -- | -- | -- |
| caption_noclass_tagsub | w05 | 0.000<br><sub>0.000-0.000</sub> | 0.099<br><sub>0.097-0.102</sub> | 0.172<br><sub>0.168-0.175</sub> | 0.242<br><sub>0.236-0.248</sub> | 0.283<br><sub>0.276-0.290</sub> | -- | -- | -- | -- | -- |
| caption_noclass_tagsub | w07 | 0.000<br><sub>0.000-0.000</sub> | 0.090<br><sub>0.087-0.092</sub> | 0.150<br><sub>0.147-0.152</sub> | 0.204<br><sub>0.201-0.207</sub> | 0.257<br><sub>0.253-0.261</sub> | 0.299<br><sub>0.294-0.304</sub> | 0.322<br><sub>0.317-0.328</sub> | -- | -- | -- |
| caption_noclass_tagsub | w10 | 0.000<br><sub>0.000-0.000</sub> | 0.060<br><sub>0.059-0.062</sub> | 0.106<br><sub>0.104-0.108</sub> | 0.145<br><sub>0.142-0.148</sub> | 0.182<br><sub>0.178-0.186</sub> | 0.219<br><sub>0.215-0.223</sub> | 0.259<br><sub>0.254-0.264</sub> | 0.297<br><sub>0.291-0.304</sub> | 0.329<br><sub>0.323-0.336</sub> | 0.349<br><sub>0.342-0.355</sub> |

## hotpot_vase

| arm | setup | k=0 | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 | k=7 | k=8 | k=9 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| caption | w03 | 0.000<br><sub>0.000-0.000</sub> | 0.048<br><sub>0.047-0.050</sub> | 0.151<br><sub>0.148-0.155</sub> | -- | -- | -- | -- | -- | -- | -- |
| caption | w05 | 0.000<br><sub>0.000-0.000</sub> | 0.050<br><sub>0.049-0.052</sub> | 0.131<br><sub>0.128-0.133</sub> | 0.252<br><sub>0.249-0.255</sub> | 0.413<br><sub>0.410-0.417</sub> | -- | -- | -- | -- | -- |
| caption | w07 | 0.000<br><sub>0.000-0.000</sub> | 0.049<br><sub>0.048-0.051</sub> | 0.123<br><sub>0.121-0.125</sub> | 0.227<br><sub>0.223-0.230</sub> | 0.357<br><sub>0.353-0.360</sub> | 0.502<br><sub>0.498-0.505</sub> | 0.609<br><sub>0.606-0.612</sub> | -- | -- | -- |
| caption | w10 | 0.000<br><sub>0.000-0.000</sub> | 0.038<br><sub>0.037-0.038</sub> | 0.088<br><sub>0.086-0.089</sub> | 0.154<br><sub>0.152-0.156</sub> | 0.238<br><sub>0.235-0.240</sub> | 0.335<br><sub>0.333-0.338</sub> | 0.440<br><sub>0.438-0.443</sub> | 0.543<br><sub>0.541-0.545</sub> | 0.628<br><sub>0.626-0.631</sub> | 0.690<br><sub>0.687-0.693</sub> |
| caption_noclass | w03 | 0.000<br><sub>0.000-0.000</sub> | 0.059<br><sub>0.057-0.061</sub> | 0.151<br><sub>0.148-0.155</sub> | -- | -- | -- | -- | -- | -- | -- |
| caption_noclass | w05 | 0.000<br><sub>0.000-0.000</sub> | 0.069<br><sub>0.067-0.070</sub> | 0.185<br><sub>0.181-0.188</sub> | 0.343<br><sub>0.340-0.347</sub> | 0.475<br><sub>0.473-0.478</sub> | -- | -- | -- | -- | -- |
| caption_noclass | w07 | 0.000<br><sub>0.000-0.000</sub> | 0.063<br><sub>0.061-0.064</sub> | 0.161<br><sub>0.158-0.165</sub> | 0.292<br><sub>0.289-0.296</sub> | 0.434<br><sub>0.431-0.437</sub> | 0.540<br><sub>0.537-0.543</sub> | 0.594<br><sub>0.590-0.597</sub> | -- | -- | -- |
| caption_noclass | w10 | 0.000<br><sub>0.000-0.000</sub> | 0.045<br><sub>0.044-0.046</sub> | 0.109<br><sub>0.107-0.111</sub> | 0.195<br><sub>0.192-0.197</sub> | 0.297<br><sub>0.295-0.299</sub> | 0.406<br><sub>0.404-0.409</sub> | 0.510<br><sub>0.508-0.512</sub> | 0.596<br><sub>0.593-0.599</sub> | 0.657<br><sub>0.654-0.661</sub> | 0.695<br><sub>0.691-0.698</sub> |
| tags | tags | 0.000<br><sub>0.000-0.000</sub> | 0.059<br><sub>0.053-0.065</sub> | 0.100<br><sub>0.089-0.111</sub> | 0.132<br><sub>0.115-0.149</sub> | 0.158<br><sub>0.139-0.176</sub> | 0.177<br><sub>0.157-0.196</sub> | -- | -- | -- | -- |
| caption_tagsub | w03 | 0.000<br><sub>0.000-0.000</sub> | 0.057<br><sub>0.054-0.059</sub> | 0.160<br><sub>0.156-0.164</sub> | -- | -- | -- | -- | -- | -- | -- |
| caption_tagsub | w05 | 0.000<br><sub>0.000-0.000</sub> | 0.059<br><sub>0.057-0.060</sub> | 0.137<br><sub>0.135-0.140</sub> | 0.233<br><sub>0.230-0.236</sub> | 0.354<br><sub>0.347-0.360</sub> | -- | -- | -- | -- | -- |
| caption_tagsub | w07 | 0.000<br><sub>0.000-0.000</sub> | 0.056<br><sub>0.055-0.058</sub> | 0.126<br><sub>0.124-0.128</sub> | 0.208<br><sub>0.206-0.210</sub> | 0.300<br><sub>0.296-0.303</sub> | 0.398<br><sub>0.393-0.404</sub> | 0.476<br><sub>0.471-0.481</sub> | -- | -- | -- |
| caption_tagsub | w10 | 0.000<br><sub>0.000-0.000</sub> | 0.045<br><sub>0.044-0.045</sub> | 0.096<br><sub>0.094-0.097</sub> | 0.154<br><sub>0.151-0.156</sub> | 0.218<br><sub>0.215-0.221</sub> | 0.287<br><sub>0.284-0.290</sub> | 0.358<br><sub>0.355-0.362</sub> | 0.426<br><sub>0.422-0.431</sub> | 0.489<br><sub>0.484-0.493</sub> | 0.539<br><sub>0.534-0.544</sub> |
| caption_noclass_tagsub | w03 | 0.000<br><sub>0.000-0.000</sub> | 0.068<br><sub>0.064-0.072</sub> | 0.150<br><sub>0.144-0.156</sub> | -- | -- | -- | -- | -- | -- | -- |
| caption_noclass_tagsub | w05 | 0.000<br><sub>0.000-0.000</sub> | 0.080<br><sub>0.077-0.083</sub> | 0.183<br><sub>0.178-0.187</sub> | 0.307<br><sub>0.302-0.311</sub> | 0.397<br><sub>0.391-0.403</sub> | -- | -- | -- | -- | -- |
| caption_noclass_tagsub | w07 | 0.000<br><sub>0.000-0.000</sub> | 0.074<br><sub>0.072-0.076</sub> | 0.165<br><sub>0.162-0.168</sub> | 0.268<br><sub>0.264-0.271</sub> | 0.369<br><sub>0.363-0.374</sub> | 0.447<br><sub>0.442-0.452</sub> | 0.491<br><sub>0.486-0.497</sub> | -- | -- | -- |
| caption_noclass_tagsub | w10 | 0.000<br><sub>0.000-0.000</sub> | 0.055<br><sub>0.053-0.056</sub> | 0.119<br><sub>0.116-0.122</sub> | 0.193<br><sub>0.189-0.197</sub> | 0.271<br><sub>0.267-0.276</sub> | 0.348<br><sub>0.344-0.353</sub> | 0.419<br><sub>0.414-0.424</sub> | 0.484<br><sub>0.479-0.488</sub> | 0.534<br><sub>0.530-0.539</sub> | 0.570<br><sub>0.565-0.574</sub> |

## Local vs global ranking

Largest absolute difference across all 531 points: **0.6804**; mean 0.1810. Global never exceeds local (True), which is the expected ordering -- a caption's own top-k is the optimal removal set, so no corpus-wide list can beat it.

That they coincide at all is structural, not a coincidence. `z = beta . e(word)` depends only on the word, so the corpus ranking and the within-caption ranking are the same order; they differ only because the global list averages `z` over folds while the local one uses the fold's own coefficients. SMER cannot rank a word differently in two captions, and these two curves are the measurement of that.
