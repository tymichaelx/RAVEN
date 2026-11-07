# TTP Data Pipeline

Builds leakage-safe train/dev/test splits for sentence-level multi-label ATT&CK technique
classification from TRAM v2, plus a document-level transfer set from AnnoCTR.

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/). `uv.lock` pins every
package version and `.python-version` pins Python, so everyone gets the same environment.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # once per machine
uv sync                                           # creates .venv/ from uv.lock
```

Add a dependency with `uv add <package>`, never by hand, and commit the updated
`pyproject.toml` and `uv.lock` together. Run scripts with `uv run python ...`, or select
`.venv/bin/python` as the interpreter in your editor.

## Run

```bash
make dataset      # clones both corpora at pinned commits, then writes data/extracted/ ; ~5 seconds
make techniques   # data/techniques.json
make eda          # eda_results/
make holdout      # data/holdout.json
```

## Evaluation

Every model is scored by `evaluation/evaluate.py`, so all conditions are compared with the
same code. A model writes one JSON line per sentence of the split (use `write_predictions`):

```json
{"doc_id": "...", "sent_idx": 3, "predicted": ["T1059"], "scores": {"T1059": 0.91, "T1027": 0.40}}
```

`predicted` is the final answer (may be empty). `scores` is optional and enables the ranking
metrics. Every sentence of the split must be present.

```bash
make evaluate PRED=runs/retriever/test.jsonl              # full run; SPLIT=dev to score dev
make evaluate PRED=runs/retriever_withheld/test.jsonl WITHHELD=1   # withheld run
make random-baseline                                      # chance-level floor for hypothesis 3
make test                                                 # checks the scoring math
```

It compares the model's guesses with the correct labels and reports:

| Metric | What it tells you |
|---|---|
| Precision | Of the techniques the model guessed, how many were right. Low = many false alarms. |
| Recall | Of the techniques that were really there, how many the model found. Low = many misses. |
| F1 | One number balancing precision and recall. High only when both are high. |
| Micro F1 | F1 over all guesses pooled together. Dominated by the frequent techniques. |
| Macro F1 | F1 per technique, then averaged. Every technique counts equally, so rare ones matter. |
| Recall@k | How often the right technique is in the model's top k guesses (k = 1, 3, 5, 10). Shows whether the model ranks well, before any cut-off is applied. |
| Average precision | One number for how well the model ranks right answers above wrong ones, across all cut-offs. 1.0 = perfect ranking. |

Precision, recall and F1 are reported per sentence (the main result) and per report (all of
a report's sentences combined; secondary). Recall@k and average precision need the optional
`scores` field.

Which technique groups are reported depends on how the model was trained:

| Run | Trained on | Groups reported | Used for |
|---|---|---|---|
| Full (default) | all 40 | all 40; the 10 holdout techniques | H1, H2 |
| Withheld (`WITHHELD=1`) | 30, without the holdout 10 | all 40; the 30 seen; the 10 withheld | H3 |

H3 compares the same 10 techniques across the two runs: trained on versus never seen.

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
