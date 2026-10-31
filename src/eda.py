import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from itertools import combinations

OUT = Path("eda_results")
OUT.mkdir(exist_ok = True)

# --- Load data
label_map = json.load(open("data/extracted/label_map.json"))
def load_jsonl(path):
        with open(path) as f:
                return [json.loads(line) for line in f if line.strip()]

split_list = ["train", "dev", "test"]
splits = {name: load_jsonl(f"data/extracted/{name}.jsonl") for name in split_list}

# --- Frequency per technique ---
freq = pd.DataFrame.from_dict(label_map, orient = "index")
freq["total"] = freq[["train", "test", "dev"]].sum(axis = 1)
table = (freq.reset_index()
.rename(columns = {"index": "technique"})
[["technique", "name", "train", "dev", "test", "total", "in_scope"]]
)
table.to_csv(OUT / "technique_frequency.csv", index = False)
table.to_markdown(OUT / "technique_frequency.md", index = False)

fig, ax = plt.subplots(figsize = (12, 5))
colors = ["tab:blue" if s else "tab:gray" for s in freq["in_scope"]]
ax.bar(freq.index, freq["train"], color = colors)
ax.axhline(20, color = "tab:red", linestyle = "--", linewidth = 1,
           label = "N = 20 scope cutoff")
ax.set_ylabel("Training sentences")
ax.set_xlabel("ATT&CK technique")
ax.set_title("Technique frequency in the training split")
ax.tick_params(axis = "x", rotation = 90, labelsize = 8)
ax.legend()
plt.tight_layout()

fig.savefig(OUT / "technique_frequency.png", dpi = 200, bbox_inches = "tight")

# --- Labeled Sentences ---
# Labeled sentences are in-scope techniques and Unlabeled are out of scope
# Our evaluation only scores the 40 in-scope techniques, so a sentence labeled
# only with an out-of-scope technique behaves like an unlabeled sentence during
# evaluation.
rows = []
for name, data in splits.items():
    n = len(data)
    labeled = sum(1 for r in data if r["labels"])
    rows.append({
        "split": name,
        "sentences": n,
        "labeled": labeled,
        "unlabeled": n - labeled,
        "labeled_pct": round((labeled / n) * 100, 1),
        "unlabeled_pct": round((n - labeled) / n * 100, 1)
    })

label_presence = pd.DataFrame(rows)
label_presence.to_csv(OUT / "label_presence.csv", index=False)
label_presence.to_markdown(OUT / "label_presence.md", index=False)

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(label_presence["split"], label_presence["labeled_pct"], label="Labeled", color="tab:blue")
ax.bar(label_presence["split"], label_presence["unlabeled_pct"],
       bottom=label_presence["labeled_pct"], label="Unlabeled", color="tab:gray")
ax.set_ylabel("Share of sentences (%)")
ax.set_title("Labeled vs. unlabeled sentences by split")
ax.set_ylim(0, 100)
ax.legend()
plt.tight_layout()
fig.savefig(OUT / "label_presence.png", dpi=200, bbox_inches="tight")

in_scope = set(freq.index[freq["in_scope"]])

for name, data in splits.items():
    in_scope_labeled = sum(1 for r in data if set(r["labels"]) & in_scope)
    print(f"{name}: {in_scope_labeled} sentences with an in-scope label "
          f"({100 * in_scope_labeled / len(data):.1f}%)")

# --- Labels per sentence ---
def bucket(n):
    return "3+" if n >= 3 else str(n)

rows = []
for name, data in splits.items():
    c = Counter(bucket(len(r["labels"])) for r in data)
    rows.append({"split": name, **{b: c.get(b, 0) for b in ("0", "1", "2", "3+")}})

labels_per_sentence = pd.DataFrame(rows).set_index("split")
labels_per_sentence["multi_label"] = labels_per_sentence[["2", "3+"]].sum(axis=1)
labels_per_sentence["multi_label_pct_of_labeled"] = (
    100 * labels_per_sentence["multi_label"]
    / labels_per_sentence[["1", "2", "3+"]].sum(axis=1)
).round(1)

labels_per_sentence.to_csv(OUT / "labels_per_sentence.csv")
labels_per_sentence.to_markdown(OUT / "labels_per_sentence.md")

ax = labels_per_sentence[["1", "2", "3+"]].plot(kind="bar", figsize=(6, 4), rot=0)
ax.set_ylabel("Sentences")
ax.set_xlabel("")
ax.set_title("Techniques per labeled sentence")
ax.legend(title="Labels")
plt.tight_layout()
ax.figure.savefig(OUT / "labels_per_sentence.png", dpi=200, bbox_inches="tight")

# --- Technique Co-occurrence ---

train = splits["train"]

pair_counts = Counter()
for r in train:
    for a, b in combinations(sorted(r["labels"]), 2):
        pair_counts[(a, b)] += 1

n_train = Counter(t for r in train for t in r["labels"])

pairs = pd.DataFrame(
    [{"tech_a": a, "tech_b": b,
      "name_a": label_map[a]["name"], "name_b": label_map[b]["name"],
      "together": n,
      "p_b_given_a": round(n / n_train[a], 2),
      "p_a_given_b": round(n / n_train[b], 2)}
     for (a, b), n in pair_counts.items()]
).sort_values("together", ascending=False)

pairs.to_csv(OUT / "cooccurrence_pairs.csv", index=False)
pairs.head(15).to_markdown(OUT / "cooccurrence_top15.md", index=False)
pairs.head(15)

techs = sorted(in_scope)
idx = {t: i for i, t in enumerate(techs)}
M = np.zeros((len(techs), len(techs)), dtype=int)
for (a, b), n in pair_counts.items():
    if a in idx and b in idx:
        M[idx[a], idx[b]] = M[idx[b], idx[a]] = n

fig, ax = plt.subplots(figsize=(11, 10))
im = ax.imshow(M, cmap="Blues")
ax.set_xticks(range(len(techs)), techs, rotation=90, fontsize=7)
ax.set_yticks(range(len(techs)), techs, fontsize=7)
ax.set_title("Sentence-level co-occurrence, in-scope techniques (train)")
fig.colorbar(im, ax=ax, shrink=0.8, label="Sentences with both labels")
plt.tight_layout()
fig.savefig(OUT / "cooccurrence_heatmap.png", dpi=200, bbox_inches="tight")

# --- Per-technique test counts for holdout selection ---

techniques = json.load(open("data/techniques.json"))["techniques"]

holdout_pool = table.copy()
holdout_pool["tactics"] = holdout_pool["technique"].map(lambda t: ", ".join(techniques[t]["tactics"]))
holdout_pool["eligible"] = holdout_pool["in_scope"] & (holdout_pool["test"] >= 5)

cols = ["technique", "name", "tactics", "train", "dev", "test", "eligible"]
holdout_pool[cols].sort_values("test", ascending=False).to_csv(OUT / "holdout_pool.csv", index=False)
holdout_pool[holdout_pool["eligible"]][cols].to_markdown(OUT / "holdout_eligible.md", index=False)

print(f"{holdout_pool['eligible'].sum()} of {holdout_pool['in_scope'].sum()} in-scope techniques are eligible")