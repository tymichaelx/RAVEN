# TTP Data Pipeline

Builds leakage-safe train/dev/test splits for sentence-level multi-label ATT&CK technique
classification from TRAM v2, plus a document-level transfer set from AnnoCTR.

## Run

```bash
bash fetch_data.sh          # clones both corpora at pinned commits into raw/
python build_dataset.py     # writes data/ ; ~5 seconds
```

Track A (TRAM-Clean released): list its report `doc_id`s one per line, then
`python build_dataset.py --holdout-docs tram_clean_ids.txt` to force them into test.

Track B (default): fill in `data/audit_sheet.csv`. For each sentence set `decision` to
`keep`, `remove`, or `add`, and put the corrected parent IDs in `corrected_labels`.
Justify every change against the official ATT&CK technique description.

## Sources

| Corpus | Commit | Role | Unit |
|---|---|---|---|
| TRAM v2 `multi_label.json` | `f29793d` | train / dev / test | sentence, 50 techniques |
| AnnoCTR `linking_mitre_only` + `text/test` | `d510b69` | held-out transfer only | document |

## Processing decisions

1. **Header removal.** TRAM prepends `title: ... url: <link>` to each report's first sentence;
   its splitter sometimes breaks the title at a colon and strands the URL in sentence 1. Both
   are stripped so vendor domains can't act as spurious features.
2. **Markdown links** `[text](url)` reduced to `text`.
3. **Overlong unlabeled "sentences"** (> 2,000 chars; IOC dumps, code) dropped. Labeled ones kept.
4. **Parent normalization.** `T1003.001` becomes `T1003`. Original labels kept in `raw_labels`.
   In TRAM every parent maps to exactly one label, so this does not shrink the label space
   here; it matters for aligning with TRAM-Clean and AnnoCTR, which both use parent IDs.
5. **Leakage grouping.** Reports sharing at least 50% of their long sentences are unioned
   and always assigned to the same split.
6. **Group-level split** (70/15/15 by report). 500 random splits are scored; the kept one
   minimizes labels absent from dev/test, then L1 gap between label distributions.
7. **Cross-split exact-duplicate removal.** Dev/test sentences whose text appears in an
   earlier split are dropped.
8. **Label scope.** A technique is in scope if it has at least N = 20 training sentences.
   Labels are not deleted; `label_map.json` carries the flag and the eval harness applies it.

**Not done here:** lowercasing, stopword removal, stemming. Those belong to the TF-IDF
feature extractor and are ablated in week 4.

## Findings worth reporting

- CISA advisory AA22-320A appears twice under different titles with ~86% sentence overlap.
  A naive report-level split can put one copy in train and one in test.
- 350 sentence strings occur in more than one report, and 15 identical sentences carry
  conflicting labels: direct evidence of the annotation noise TRAM-Clean was built to fix.
- AnnoCTR labels technique *mentions*; TRAM labels sentences that *describe* a technique.
  Transfer results measure shift in annotation semantics as well as in source style.

## Outputs (`data/`)

`train|dev|test.jsonl`, `annoctr_test.jsonl`, `label_map.json`, `split_manifest.json`,
`audit_sheet.csv`, `stats.json`. See the docstring in `build_dataset.py` for schemas.
