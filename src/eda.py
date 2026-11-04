import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from itertools import combinations
import string
import nltk
from nltk import FreqDist
from nltk import ConditionalFreqDist
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer

OUT = Path("eda_results")
OUT.mkdir(exist_ok = True)

# --- Load data
label_map = json.load(open("data/extracted/label_map.json"))


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


split_list = ["train", "dev", "test"]
splits = {name: load_jsonl(f"data/extracted/{name}.jsonl") for name in
          split_list}

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
label_presence.to_csv(OUT / "label_presence.csv", index = False)
label_presence.to_markdown(OUT / "label_presence.md", index = False)

fig, ax = plt.subplots(figsize = (6, 4))
ax.bar(label_presence["split"], label_presence["labeled_pct"],
       label = "Labeled", color = "tab:blue")
ax.bar(label_presence["split"], label_presence["unlabeled_pct"],
       bottom = label_presence["labeled_pct"], label = "Unlabeled",
       color = "tab:gray")
ax.set_ylabel("Share of sentences (%)")
ax.set_title("Labeled vs. unlabeled sentences by split")
ax.set_ylim(0, 100)
ax.legend()
plt.tight_layout()
fig.savefig(OUT / "label_presence.png", dpi = 200, bbox_inches = "tight")

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
    rows.append(
        {"split": name, **{b: c.get(b, 0) for b in ("0", "1", "2", "3+")}})

labels_per_sentence = pd.DataFrame(rows).set_index("split")
labels_per_sentence["multi_label"] = labels_per_sentence[["2", "3+"]].sum(
    axis = 1)
labels_per_sentence["multi_label_pct_of_labeled"] = (
        100 * labels_per_sentence["multi_label"]
        / labels_per_sentence[["1", "2", "3+"]].sum(axis = 1)
).round(1)

labels_per_sentence.to_csv(OUT / "labels_per_sentence.csv")
labels_per_sentence.to_markdown(OUT / "labels_per_sentence.md")

ax = labels_per_sentence[["1", "2", "3+"]].plot(kind = "bar", figsize = (6, 4),
                                                rot = 0)
ax.set_ylabel("Sentences")
ax.set_xlabel("")
ax.set_title("Techniques per labeled sentence")
ax.legend(title = "Labels")
plt.tight_layout()
ax.figure.savefig(OUT / "labels_per_sentence.png", dpi = 200,
                  bbox_inches = "tight")

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
).sort_values("together", ascending = False)

pairs.to_csv(OUT / "cooccurrence_pairs.csv", index = False)
pairs.head(15).to_markdown(OUT / "cooccurrence_top15.md", index = False)
pairs.head(15)

techs = sorted(in_scope)
idx = {t: i for i, t in enumerate(techs)}
M = np.zeros((len(techs), len(techs)), dtype = int)
for (a, b), n in pair_counts.items():
    if a in idx and b in idx:
        M[idx[a], idx[b]] = M[idx[b], idx[a]] = n

fig, ax = plt.subplots(figsize = (11, 10))
im = ax.imshow(M, cmap = "Blues")
ax.set_xticks(range(len(techs)), techs, rotation = 90, fontsize = 7)
ax.set_yticks(range(len(techs)), techs, fontsize = 7)
ax.set_title("Sentence-level co-occurrence, in-scope techniques (train)")
fig.colorbar(im, ax = ax, shrink = 0.8, label = "Sentences with both labels")
plt.tight_layout()
fig.savefig(OUT / "cooccurrence_heatmap.png", dpi = 200, bbox_inches = "tight")

# --- Per-technique test counts for holdout selection ---

techniques = json.load(open("data/techniques.json"))["techniques"]

holdout_pool = table.copy()
holdout_pool["tactics"] = holdout_pool["technique"].map(
    lambda t: ", ".join(techniques[t]["tactics"]))
holdout_pool["eligible"] = holdout_pool["in_scope"] & (
            holdout_pool["test"] >= 5)

cols = ["technique", "name", "tactics", "train", "dev", "test", "eligible"]
holdout_pool[cols].sort_values("test", ascending = False).to_csv(
    OUT / "holdout_pool.csv", index = False)
holdout_pool[holdout_pool["eligible"]][cols].to_markdown(
    OUT / "holdout_eligible.md", index = False)

print(
    f"{holdout_pool['eligible'].sum()} of {holdout_pool['in_scope'].sum()} in-scope techniques are eligible")

# --- Normalization for analysis only ---
nltk.download("punkt_tab")
nltk.download("stopwords")

train_text = [r["text"] for r in splits["train"]]
labeled_text = [r["text"] for r in splits["train"] if r["labels"]]

STOP = set(stopwords.words("english"))
PUNCT = set(string.punctuation) | {"``", "''", "’", "“", "”", "--", "..."}


def raw_tokens(texts):
    return [t.lower() for s in texts for t in word_tokenize(s)]


def clean_tokens(texts):
    return [t for t in raw_tokens(texts)
            if t not in STOP and t not in PUNCT and not all(
            c in string.punctuation for c in t)]


fd_raw = FreqDist(raw_tokens(train_text))
fd_clean = FreqDist(clean_tokens(train_text))


def plot_top(fd, title, filename, n = 20):
    words, counts = zip(*fd.most_common(n))
    fig, ax = plt.subplots(figsize = (8, 4))
    ax.bar(words, counts)
    ax.set_title(title)
    ax.set_ylabel("Count")
    ax.tick_params(axis = "x", rotation = 60)
    plt.tight_layout()
    fig.savefig(OUT / filename, dpi = 200, bbox_inches = "tight")
    plt.show()


plot_top(fd_raw, "Top 20 tokens, raw (train)", "freq_top20_raw.png")
plot_top(fd_clean, "Top 20 tokens, stopwords and punctuation removed (train)",
         "freq_top20_clean.png")

pd.DataFrame({
    "raw_token": [w for w, _ in fd_raw.most_common(20)],
    "raw_count": [c for _, c in fd_raw.most_common(20)],
    "clean_token": [w for w, _ in fd_clean.most_common(20)],
    "clean_count": [c for _, c in fd_clean.most_common(20)],
}).to_markdown(OUT / "freq_top20.md", index = False)


# --- Word clouds
def save_cloud(fd, title, filename):
    wc = WordCloud(width = 1200, height = 600, background_color = "white",
                   max_words = 150)
    wc.generate_from_frequencies(dict(fd))
    fig, ax = plt.subplots(figsize = (12, 6))
    ax.imshow(wc, interpolation = "bilinear")
    ax.axis("off")
    ax.set_title(title)
    fig.savefig(OUT / filename, dpi = 200, bbox_inches = "tight")
    plt.show()


save_cloud(fd_clean, "All training sentences", "wordcloud_all.png")
save_cloud(FreqDist(clean_tokens(labeled_text)),
           "Labeled training sentences only", "wordcloud_labeled.png")

cfd = ConditionalFreqDist(
    (tech, tok)
    for r in splits["train"] if r["labels"]
    for tok in clean_tokens([r["text"]])
    for tech in r["labels"]
)

for tech in ("T1003", "T1055"):
    print(tech, label_map[tech]["name"])
    print("  ", cfd[tech].most_common(10))

techs = sorted(in_scope)
docs = [" ".join(cfd[t].elements()) for t in techs]

vec = TfidfVectorizer(token_pattern = r"\S+", sublinear_tf = True)
X = vec.fit_transform(docs)
vocab = np.array(vec.get_feature_names_out())

rows = []
for i, t in enumerate(techs):
    weights = X[i].toarray().ravel()
    top = weights.argsort()[::-1][:10]
    rows.append({"technique": t, "name": label_map[t]["name"],
                 "top_raw": ", ".join(w for w, _ in cfd[t].most_common(10)),
                 "top_distinctive": ", ".join(vocab[top])})

cond_freq = pd.DataFrame(rows)
cond_freq.to_csv(OUT / "conditional_frequency.csv", index = False)
cond_freq.to_markdown(OUT / "conditional_frequency.md", index = False)
cond_freq[cond_freq["technique"].isin(["T1003", "T1055"])]

nltk.download("wordnet", quiet = True)

TOKENIZERS = {
    "MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
    "SecureBERT": "ehsanaghaei/SecureBERT",
}

SENT_MAX = 128
DESC_MAX = 256
PAIR_MAX = 512

train = splits["train"]
sentence_txt = [r["text"] for r in train]
techniques_txt = {t: techniques[t]["text"] for t in sorted(in_scope)}


def token_lengths(tokenizer, texts):
    """Token counts including special token ([CLS]/[SEP] or <s>/</s>"""
    return np.array([len(ids) for ids in
                     tokenizer(list(texts), add_special_tokens = True)[
                         "input_ids"]])


def length_summary(lengths, cutoff):
    return {
        "median": int(np.median(lengths)),
        "p95": int(np.percentile(lengths, 95)),
        "p99": int(np.percentile(lengths, 99)),
        "max": int(np.max(lengths)),
        "cutoff": cutoff,
        "pct_within_cutoff": round(100 * (lengths <= cutoff).mean(), 2),
        "n_truncated": int((lengths > cutoff).sum()),
    }


def plot_lengths(lengths_by_model, cutoff, title, filename, bins = 60):
    fig, ax = plt.subplots(figsize = (8, 4))
    clip = max(cutoff * 2, 1)
    for name, lengths in lengths_by_model.items():
        ax.hist(np.minimum(lengths, clip), bins = bins, alpha = 0.6,
                label = name)
    ax.axvline(cutoff, color = "tab:red", linestyle = "--",
               label = f"Cutoff = {cutoff}")
    ax.set_xlabel(f"Tokens (values above {clip} shown at {clip})")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUT / filename, dpi = 200, bbox_inches = "tight")
    plt.show()


# ----- Sentence and technique text lengths -----
tokenizers = {name: AutoTokenizer.from_pretrained(path) for name, path in
              TOKENIZERS.items()}

sentence_len = {name: token_lengths(tok, sentence_txt) for name, tok in
                tokenizers.items()}
desc_lengths = {name: token_lengths(tok, techniques_txt.values()) for name, tok
                in tokenizers.items()}

length_table = pd.DataFrame(
    [{"input": "sentence", "tokenizer": n, **length_summary(l, SENT_MAX)} for
     n, l in sentence_len.items()] +
    [{"input": "technique text", "tokenizer": n, **length_summary(l, DESC_MAX)}
     for n, l in desc_lengths.items()]
)

length_table.to_csv(OUT / "token_lengths.csv", index = False)
length_table.to_markdown(OUT / "token_lengths.md", index = False)
print(length_table)

plot_lengths(sentence_len, SENT_MAX, "Training sentence length in tokens",
             "token_sentences.png")
plot_lengths(desc_lengths, DESC_MAX,
             "Technique text length in tokens (in-scope)",
             "token_lengths_techniques.png", bins = 20)

# ----- Verifier pair length: [CLS] sentence [SEP] technique text -----
# The cross-encoder reads both together, so their combined length must fit the model.
secure_BERT = tokenizers["SecureBERT"]
pair_lengths = []
for r in train:
    for t in r["labels"]:
        if t in techniques_txt:
            pair_lengths.append(len(secure_BERT(r["text"], techniques_txt[t],
                                                add_special_tokens = True)[
                                        "input_ids"]))
pair_lengths = np.array(pair_lengths)
pair_summary = length_summary(pair_lengths, PAIR_MAX)
print("Verifier pair lengths (SecureBERT, gold pairs):", pair_summary)
pd.DataFrame([pair_summary]).to_markdown(OUT / "token_lengths_pairs.md",
                                         index = False)

# ----- Lexical overlap between sentences and their gold technique text -----
# We perform lemmatization to ensure technique matching comes from different
# vocabulary and not overlap in word forms
lemmatizer = WordNetLemmatizer()


def content_words(text):
    """Lowercased, lemmatized tokens with stopwords and punctuation removed."""
    return {
        lemmatizer.lemmatize(t)
        for t in (w.lower() for w in word_tokenize(text))
        if t not in STOP and t not in PUNCT and not all(
            c in string.punctuation for c in t)
    }


def jaccard(a, b):
    """
    Used as a measure the overlap between a sentence's content W_s and a
    technique's description W_t
    """
    return len(a & b) / len(a | b) if a | b else 0.0


tech_words = {t: content_words(txt) for t, txt in techniques_txt.items()}

rows = []
for r in train:
    gold = [t for t in r["labels"] if t in tech_words]
    if not gold:
        continue
    words = content_words(r["text"])
    scores = {t: jaccard(words, tw) for t, tw in tech_words.items()}
    ranking = sorted(scores, key = scores.get, reverse = True)
    for t in gold:
        rows.append({
            "doc_id": r["doc_id"], "sent_idx": r["sent_idx"], "technique": t,
            "jaccard_gold": scores[t],
            "shared_words": len(words & tech_words[t]),
            "gold_rank": ranking.index(t) + 1,
            # 1 = gold technique has the highest overlap
        })

overlap = pd.DataFrame(rows)
overlap.to_csv(OUT / "lexical_overlap_pairs.csv", index = False)

overlap_summary = {
    "gold_pairs": len(overlap),
    "median_jaccard": round(overlap["jaccard_gold"].median(), 3),
    "pct_zero_overlap": round(100 * (overlap["shared_words"] == 0).mean(), 1),
    "pct_gold_ranked_1": round(100 * (overlap["gold_rank"] == 1).mean(), 1),
    "pct_gold_in_top5": round(100 * (overlap["gold_rank"] <= 5).mean(), 1),
    "median_gold_rank": int(overlap["gold_rank"].median()),
    "candidates": len(tech_words),
}
pd.DataFrame([overlap_summary]).to_markdown(OUT / "lexical_overlap_summary.md",
                                            index = False)
print("Lexical overlap:", overlap_summary)

per_tech = (overlap.groupby("technique")
            .agg(pairs = ("jaccard_gold", "size"),
                 median_jaccard = ("jaccard_gold", "median"),
                 pct_ranked_1 = ("gold_rank",
                                 lambda s: round(100 * (s == 1).mean(), 1)))
            .sort_values("median_jaccard"))
per_tech.insert(0, "name", per_tech.index.map(lambda t: techniques[t]["name"]))
per_tech.to_markdown(OUT / "lexical_overlap_by_technique.md")

fig, axes = plt.subplots(1, 2, figsize = (12, 4))
axes[0].hist(overlap["jaccard_gold"], bins = 40)
axes[0].set_xlabel("Jaccard similarity, sentence vs. gold technique text")
axes[0].set_ylabel("Gold (sentence, technique) pairs")
axes[0].set_title("Lexical overlap with the correct technique")
axes[1].hist(overlap["gold_rank"], bins = range(1, len(tech_words) + 2),
             align = "left")
axes[1].set_xlabel(f"Rank of gold technique by Jaccard (of {len(tech_words)})")
axes[1].set_ylabel("Gold pairs")
axes[1].set_title("Would keyword overlap find the right technique?")
plt.tight_layout()
fig.savefig(OUT / "lexical_overlap.png", dpi = 200, bbox_inches = "tight")
plt.show()
