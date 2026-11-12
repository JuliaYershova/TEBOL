# User study: answers behind the paper's numbers

`ratings.csv` holds one row per participant (270) and nothing that identifies anyone:
no respondent id, date, time, recruitment group, seed or free text. It is cut from the
private LimeSurvey export by `scripts/survey_anonymize.py`.

| column | meaning |
|---|---|
| `participant` | row of the export, 0-based; the key `response_row` in `data/manifest/survey_comment_codes.csv` |
| `language` | language the participant chose: `cs`, `en`, `uk` |
| `expertise` | machine-learning experience: `none`, `beginner`, `intermediate or advanced` |
| `attention_ok` | `True` if all three attention-check questions were answered as instructed (2 participants failed) |
| `comprehension_bb`, `comprehension_ht` | final question, how easy the format was to comprehend overall: 1 Very difficult ... 5 Very easy |
| `<class>_bb`, `<class>_ht` | how well the display captures the named object: 1 Very Poor ... 5 Excellent, one per class and format |

`bb` is the bounding box, `ht` the Grad-CAM heatmap. Every participant rated both formats
for all ten classes, so each comparison is paired within the participant.

```
python scripts/survey_stats.py
```

writes `results/metrics/survey_stats.csv` and `survey_counts.csv`, the source of every
user-study number in the paper: the means 4.06 and 3.49 and their Wilcoxon and paired t
tests (`comprehension`), the pooled and per-class means of the capture ratings with
Bonferroni-corrected Wilcoxon tests (`capture`), and the counts of the coded comments.
The paper's numbers use all 270 participants. The rows `sensitivity: ...` repeat the two
overall comparisons without the participants who failed an attention check.

## Coded comments

The comments themselves are not published. `data/manifest/survey_comment_codes.csv`
holds, for each non-empty comment, its sentiment and themes as coded by ChatGPT
(sentiment: positive, negative, mixed, neutral; themes: C clarity, A accuracy,
B boundaries, O image hidden under the overlay, K colours).
