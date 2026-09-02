# Class-name leakage in the captions

How often a caption names the class of the image it describes. A caption
that contains its own label makes classification trivial and any
explanation of that classification circular, so the rate is reported here
and the words are removed for the ablated rerun.

## Method

**Term list.** `data/manifest/class_terms.csv`, fixed in advance, three
tiers:

| tier | what |
|---|---|
| 1 | the class name and its morphological variants -- plural, possessive, hyphenated and open-compound spellings |
| 2 | the agreed synonyms for that class, and their variants |
| 3 | the standalone head noun of a two-word class name (`guitar`, `pot`, `truck`) |

The tier-2 synonyms were fixed by the authors rather than derived, one
list per class:

| class | synonyms |
|---|---|
| hot pot | steamboat, Chinese fondue |
| vase | flower vase, vessel, urn |
| ambulance | emergency vehicle, medical transport, rescue vehicle |
| fire truck | fire apparatus, fire vehicle |
| cucumber | cuke |
| zucchini | courgette, baby marrow |
| bee | honeybee, bumblebee, pollinator |
| ant | formicid, social insect |
| acoustic guitar | unplugged guitar, steel-string guitar, classical guitar |
| violin | fiddle |

**Why the list is fixed rather than mined.** Automatic expansion was tried
and rejected. Embedding neighbours of the class lemma fail on short tokens,
which cluster by spelling rather than meaning -- the nearest neighbours of
`ant` are `and`, `at`, `against`, `an` at cosine 0.85-0.89. Asking a
language model to label corpus candidates failed in both directions: under
one rubric it called `stew` and `soup` components rather than names of a
hot pot, under another it called `flowers`, `nectar` and `pollen` names for
a bee. A fixed list makes the audit reproducible and puts the judgement
where a reviewer can see it.

**One deliberate divergence from WordNet.** The ImageNet lemma for
`n03345487` is *"fire engine, fire truck"*, putting both in one synset.
They are not the same apparatus: an engine is a pumper, carrying water and
hoses, while a truck is an aerial, carrying ladders. `fire engine` is
therefore excluded from the term list even though the dataset's own lemma
contains it. The cost is measurable and small -- 751 captions (2.8% of the
class) name the class only that way, moving the class rate from 97.1% to
94.3%. No caption of the contrasting class uses the phrase at all.

**Tier 3 and the two-word classes.** Three classes have two-word names, and
captions use the head noun alone: `pot` without `hot`, `truck` without
`fire`, `guitar` without `acoustic`. Tier 3 catches those. Modifiers alone
(`fire`, `hot`, `acoustic`) are *not* removed, because they routinely
describe the scene rather than the object.

**Substring matching was rejected outright**: `ant` is a substring of
`vibrant` (16,222 occurrences), `plant`, `elegant`, `cilantro`; `bee` of
`beef` (1,775), `beer`, `beetle`.

**Matching rule.** Token-level, on the stage-1 tokenisation. A token
counts if it equals a term, if any of its hyphen-separated parts does
(`tofu-hotpot`, `vegetables-zucchini` -- stage 1 ran with
`split_hyphens=False`), or after stripping a possessive `'s`. Multiword
lemmas (`fire truck`, `hot pot`, `acoustic guitar`) match a consecutive
run of tokens.

**Scope.** Tiers 1-3 applied. Only a caption's *own* class terms
count; a cucumber caption that says `zucchini` is not counted here.

## Captions naming their own class

| pair | class | captions | naming the class | % | tokens removed | % of tokens |
|---|---|---|---|---|---|---|
| acousticguitar_violin | acousticguitar | 80,680 | 79,127 | 98.1% | 142,111 | 11.1% |
| acousticguitar_violin | violin | 53,200 | 46,892 | 88.1% | 47,560 | 5.5% |
| ambulance_firetruck | ambulance | 60,400 | 54,655 | 90.5% | 61,881 | 6.2% |
| ambulance_firetruck | firetruck | 54,239 | 50,322 | 92.8% | 94,084 | 10.2% |
| ant_bee | ant | 66,240 | 60,376 | 91.1% | 63,972 | 6.1% |
| ant_bee | bee | 66,879 | 63,143 | 94.4% | 67,205 | 6.5% |
| cucumber_zucchini | cucumber | 50,720 | 38,690 | 76.3% | 40,036 | 5.2% |
| cucumber_zucchini | zucchini | 43,719 | 40,324 | 92.2% | 41,607 | 6.2% |
| hotpot_vase | hotpot | 45,960 | 26,462 | 57.6% | 43,451 | 5.7% |
| hotpot_vase | vase | 51,999 | 40,278 | 77.5% | 41,268 | 5.0% |
| **TOTAL** |  | **574,036** | **500,269** | **87.1%** | **643,175** | **7.0%** |

## By caption length

| requested words | captions | naming the class | % | mean length | after removal | empty after |
|---|---|---|---|---|---|---|
| 3 | 71,755 | 56,716 | 79.0% | 3.71 | 2.73 | 14 |
| 5 | 71,755 | 61,601 | 85.8% | 5.70 | 4.63 | 0 |
| 7 | 71,755 | 62,361 | 86.9% | 7.22 | 6.13 | 0 |
| 10 | 71,755 | 63,350 | 88.3% | 10.31 | 9.19 | 0 |
| 15 | 71,752 | 63,691 | 88.8% | 16.76 | 15.64 | 0 |
| 20 | 71,755 | 63,945 | 89.1% | 22.73 | 21.58 | 0 |
| 25 | 71,755 | 64,044 | 89.3% | 27.31 | 26.13 | 0 |
| 30 | 71,754 | 64,561 | 90.0% | 34.40 | 33.12 | 0 |

## Limitations

- The list is curated, so it is a judgement, not a measurement. It is
  versioned in `data/manifest/class_terms.csv` so the judgement is
  auditable and the analysis rerunnable under a different one.
- Tier 3 was found by embedding search over this corpus. A language
  present only in captions not yet generated would be missed.
- Removal deletes the class word but not everything that implies it:
  `siren`, `stethoscope`, `fretboard`, `soundhole` remain, and a
  classifier can still exploit them. This bounds leakage from naming,
  not leakage in general.
- Hyponyms are treated as naming the class (`bumblebee` -> `bee`),
  which is why tier 2 moves the `bee` rate substantially. Whether that
  is the right call is a reviewer-facing choice, not a technical one.

