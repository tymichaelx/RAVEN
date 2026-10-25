"""Build leakage-safe train/dev/test splits from TRAM v2, plus an AnnoCTR transfer set.

Outputs (in --out):
  train.jsonl, dev.jsonl, test.jsonl   one sentence per line:
      {doc_id, sent_idx, text, labels, raw_labels, source}
  annoctr_test.jsonl                   one document per line:
      {doc_id, sentences, gold_all, gold_in_scope, source}
  label_map.json                       per-technique name, split counts, in_scope flag
  split_manifest.json                  doc_id -> split, plus merged leakage groups
  audit_sheet.csv                      Track B: test-split sentences for manual label audit
  stats.json                           everything you need for the data section of the report

Deliberately NOT done here: lowercasing, stopword removal, stemming. Those are
feature-extraction choices for the TF-IDF tier and get ablated in week 4.
"""
import argparse, collections, csv, json, random, re
from pathlib import Path

# TRAM prepends "title: ... url: <link>" to each report's first sentence, and its
# splitter sometimes breaks the title at a colon, stranding "url: <link>" in sentence 1.
URL_HEADER = re.compile(r"^.{0,300}?\burl:\s*https?://\S+\s*", re.I | re.S)
TITLE_TAG = re.compile(r"^\s*title:\s*", re.I)
MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
TECH_ID = re.compile(r"(T\d{4})")


def clean(text):
    """Strip TRAM's title/url header and markdown link targets; collapse whitespace."""
    text = TITLE_TAG.sub("", URL_HEADER.sub("", text))
    text = MD_LINK.sub(r"\1", text)
    return " ".join(text.split())


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:80]


def load_tram(path, max_chars):
    rows, dropped, idx = [], collections.Counter(), collections.Counter()
    for r in json.load(open(path)):
        doc, text = slug(r["doc_title"]), clean(r["sentence"])
        labels = sorted(
            {TECH_ID.match(l).group(1) for l in r["labels"]})  # parent IDs
        if not text:
            dropped["empty"] += 1
            continue
        if len(text) > max_chars and not labels:  # IOC dumps / code blobs
            dropped["overlong_unlabeled"] += 1
            continue
        rows.append(dict(doc_id = doc, sent_idx = idx[doc], text = text,
                         labels = labels,
                         raw_labels = r["labels"], source = "tram"))
        idx[doc] += 1
    return rows, dropped


def leak_groups(rows, min_len = 40, thresh = 0.5):
    """Union reports that share >= thresh of their (long) sentences, e.g. the same
    advisory republished under two titles. Grouped reports always land in one split."""
    sents = collections.defaultdict(set)
    for r in rows:
        if len(r["text"]) >= min_len:
            sents[r["doc_id"]].add(r["text"].lower())
    parent = {r["doc_id"]: r["doc_id"] for r in rows}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    docs = list(sents)
    for i, a in enumerate(docs):
        for b in docs[i + 1:]:
            shared = len(sents[a] & sents[b])
            if shared and shared / min(len(sents[a]), len(sents[b])) >= thresh:
                parent[find(a)] = find(b)
    return {d: find(d) for d in parent}


def choose_split(rows, groups, dev_frac, test_frac, n_seeds, forced_test):
    """Random group-level split; keep the seed whose dev/test label distributions
    best match train (fewest labels missing from dev/test, then smallest L1 gap)."""
    g_labels = collections.defaultdict(collections.Counter)
    for r in rows:
        g_labels[groups[r["doc_id"]]].update(r["labels"])
    all_labels = sorted({l for c in g_labels.values() for l in c})
    forced = {groups[d] for d in forced_test}
    free = sorted(set(groups.values()) - forced)
    n = len(free) + len(forced)
    n_test = max(round(test_frac * n) - len(forced), 0)
    n_dev = round(dev_frac * n)

    def dist(gs):
        c = collections.Counter()
        for g in gs:
            c.update(g_labels[g])
        tot = sum(c.values()) or 1
        return c, {l: c[l] / tot for l in all_labels}

    best = None
    for seed in range(n_seeds):
        gs = free[:]
        random.Random(seed).shuffle(gs)
        test, dev, train = set(gs[:n_test]) | forced, set(
            gs[n_test:n_test + n_dev]), set(gs[n_test + n_dev:])
        _, p_tr = dist(train)
        c_dv, p_dv = dist(dev)
        c_te, p_te = dist(test)
        missing = sum(1 for l in all_labels if c_dv[l] == 0 or c_te[l] == 0)
        l1 = sum(
            abs(p_tr[l] - p_dv[l]) + abs(p_tr[l] - p_te[l]) for l in all_labels)
        if best is None or (missing, l1) < best[0]:
            best = ((missing, l1), seed, {**{g: "train" for g in train},
                                          **{g: "dev" for g in dev},
                                          **{g: "test" for g in test}})
    return best


def technique_names(stix_path):
    names = {}
    for o in json.load(open(stix_path))["objects"]:
        if o.get("type") == "attack-pattern":
            for ref in o.get("external_references", []):
                if ref.get("source_name") == "mitre-attack":
                    names[ref["external_id"]] = o["name"]
    return names


def load_annoctr(root, split, in_scope):
    gold = collections.defaultdict(set)
    for line in open(root / "linking_mitre_only" / f"{split}.jsonl"):
        if line.strip():
            r = json.loads(line)
            m = re.search(r"/techniques/(T\d{4})", r["label_link"])
            if m:
                gold[r["document"]].add(m.group(1))
    docs = []
    for f in sorted((root / "text" / split).glob("*.txt")):
        g = sorted(gold.get(f.stem, ()))
        docs.append(dict(doc_id = f.stem, source = "annoctr",
                         sentences = [clean(l) for l in
                                      f.read_text().splitlines() if l.strip()],
                         gold_all = g,
                         gold_in_scope = [t for t in g if t in in_scope]))
    return docs


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser(description = __doc__,
                                 formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default = "raw", type = Path)
    ap.add_argument("--out", default = "extracted", type = Path)
    ap.add_argument("--dev-frac", type = float, default = 0.15)
    ap.add_argument("--test-frac", type = float, default = 0.15)
    ap.add_argument("--seeds", type = int, default = 500,
                    help = "candidate splits to score")
    ap.add_argument("--min-train", type = int, default = 20,
                    help = "N: in-scope if >= N train sentences")
    ap.add_argument("--max-chars", type = int, default = 2000)
    ap.add_argument("--leak-thresh", type = float, default = 0.5)
    ap.add_argument("--holdout-docs", type = Path,
                    help = "Track A: file of doc_ids forced into test")
    ap.add_argument("--audit-docs", type = int, default = 15,
                    help = "Track B: test docs to export for audit")
    a = ap.parse_args()
    a.out.mkdir(parents = True, exist_ok = True)

    rows, dropped = load_tram(a.raw / "tram/data/tram2-data/multi_label.json",
                              a.max_chars)
    groups = leak_groups(rows, thresh = a.leak_thresh)
    forced = a.holdout_docs.read_text().split() if a.holdout_docs else []
    (score, seed, g_split) = choose_split(rows, groups, a.dev_frac, a.test_frac,
                                          a.seeds, forced)
    doc_split = {d: g_split[g] for d, g in groups.items()}

    # Exact-duplicate removal across splits (boilerplate like "Figure 1." and reused prose).
    splits, seen, dup = {"train": [], "dev": [],
                         "test": []}, set(), collections.Counter()
    for r in rows:
        if doc_split[r["doc_id"]] == "train":
            splits["train"].append(r)
            seen.add(r["text"].lower())
    for name in ("dev", "test"):
        for r in rows:
            if doc_split[r["doc_id"]] == name:
                if r["text"].lower() in seen:
                    dup[name] += 1
                    continue
                splits[name].append(r)
        seen |= {r["text"].lower() for r in splits[name]}
    for name, rs in splits.items():
        write_jsonl(a.out / f"{name}.jsonl", rs)

    # Label map: counts per split, ATT&CK names, in-scope flag.
    names = technique_names(a.raw / "tram/data/attack/enterprise-attack.json")
    counts = {n: collections.Counter(l for r in rs for l in r["labels"]) for
              n, rs in splits.items()}
    label_map = {
        l: dict(name = names.get(l, "?"), **{n: counts[n][l] for n in splits},
                in_scope = counts["train"][l] >= a.min_train)
        for l in sorted(set().union(*counts.values()))}
    json.dump(label_map, open(a.out / "label_map.json", "w"), indent = 1)
    in_scope = {l for l, v in label_map.items() if v["in_scope"]}

    merged = collections.defaultdict(list)
    for d, g in groups.items():
        merged[g].append(d)
    json.dump(dict(seed = seed, doc_split = doc_split,
                   merged_groups = [sorted(v) for v in merged.values() if
                                    len(v) > 1]),
              open(a.out / "split_manifest.json", "w"), indent = 1)

    # AnnoCTR transfer set: document-level gold, never trained on.
    anno = load_annoctr(a.raw / "annoctr/AnnoCTR", "test", in_scope)
    write_jsonl(a.out / "annoctr_test.jsonl", anno)

    # Track B audit sheet: a seeded sample of test documents, one row per sentence.
    test_docs = sorted({r["doc_id"] for r in splits["test"]})
    audit = set(
        random.Random(0).sample(test_docs, min(a.audit_docs, len(test_docs))))
    with open(a.out / "audit_sheet.csv", "w", newline = "") as f:
        w = csv.writer(f)
        w.writerow(["doc_id", "sent_idx", "text", "tram_labels", "decision",
                    "corrected_labels", "note"])
        for r in splits["test"]:
            if r["doc_id"] in audit:
                w.writerow([r["doc_id"], r["sent_idx"], r["text"],
                            " ".join(r["labels"]), "", "", ""])

    def summ(rs):
        return dict(docs = len({r["doc_id"] for r in rs}), sentences = len(rs),
                    labeled = sum(1 for r in rs if r["labels"]),
                    label_instances = sum(len(r["labels"]) for r in rs))

    tot_gold = sum(len(d["gold_all"]) for d in anno)
    stats = dict(
        split_seed = seed,
        split_score = dict(labels_missing_from_dev_or_test = score[0],
                           l1_gap = round(score[1], 4)),
        dropped = dict(dropped), cross_split_duplicates_removed = dict(dup),
        merged_leakage_groups = sum(1 for v in merged.values() if len(v) > 1),
        splits = {n: summ(rs) for n, rs in splits.items()},
        labels_total = len(label_map), labels_in_scope = len(in_scope),
        min_train = a.min_train,
        test_label_coverage = round(
            sum(counts["test"][l] for l in in_scope) / max(
                sum(counts["test"].values()), 1), 4),
        annoctr = dict(docs = len(anno), docs_with_in_scope_gold = sum(
            1 for d in anno if d["gold_in_scope"]),
                       gold_coverage = round(
                           sum(len(d["gold_in_scope"]) for d in anno) / max(
                               tot_gold, 1), 4)),
        audit_docs = sorted(audit),
    )
    json.dump(stats, open(a.out / "stats.json", "w"), indent = 1)
    print(json.dumps(stats, indent = 1))


if __name__ == "__main__":
    main()
