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
| acousticguitar_violin | acousticguitar | 40,340 | 39,097 | 96.9% | 68,115 | 25.3% |
| acousticguitar_violin | violin | 26,600 | 22,422 | 84.3% | 22,480 | 12.8% |
| ambulance_firetruck | ambulance | 30,200 | 27,246 | 90.2% | 28,342 | 13.4% |
| ambulance_firetruck | firetruck | 27,120 | 25,574 | 94.3% | 48,591 | 25.6% |
| ant_bee | ant | 33,120 | 30,162 | 91.1% | 30,320 | 14.1% |
| ant_bee | bee | 33,440 | 31,624 | 94.6% | 31,670 | 15.0% |
| cucumber_zucchini | cucumber | 25,360 | 17,427 | 68.7% | 17,486 | 10.4% |
| cucumber_zucchini | zucchini | 21,860 | 19,654 | 89.9% | 19,721 | 13.7% |
| hotpot_vase | hotpot | 22,980 | 11,067 | 48.2% | 17,806 | 10.8% |
| hotpot_vase | vase | 26,000 | 19,755 | 76.0% | 19,829 | 10.9% |
| **TOTAL** |  | **287,020** | **244,028** | **85.0%** | **304,360** | **15.8%** |

## By caption length

| requested words | captions | naming the class | % | mean length | after removal | empty after |
|---|---|---|---|---|---|---|
| 3 | 71,755 | 56,716 | 79.0% | 3.71 | 2.73 | 14 |
| 5 | 71,755 | 61,601 | 85.8% | 5.70 | 4.63 | 0 |
| 7 | 71,755 | 62,361 | 86.9% | 7.22 | 6.13 | 0 |
| 10 | 71,755 | 63,350 | 88.3% | 10.31 | 9.19 | 0 |

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

