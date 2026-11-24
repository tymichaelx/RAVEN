"""Reference baseline: a local LLM (Ollama) labels each sentence zero-shot.

Not part of a hypothesis. It shows how a general instruction-tuned model does on the
same task with no training, scored by the same evaluate.py as the retriever and verifier.

The model reads a catalog of the 40 in-scope techniques (id, name, first sentence of
the description) and returns, for one sentence, the techniques it describes with a
confidence each. Output is constrained to a JSON schema whose ids are the 40 in-scope ids.

Decision rule, like the retriever's: predict a technique when its confidence is at least
theta, with theta tuned on dev. The LLM's confidences are coarse (mostly 0.8 or 0.9), so
untuned predictions (every technique the model names) are written too.

The LLM is never trained, so one set of predictions serves both run types. Score test
with --withheld-run to get the seen / withheld breakdown for hypothesis 3.

Usage:
  python -m src.llm_baseline predict [--model llama3.1:8b] [--workers 1]
  python -m src.llm_baseline predict --smoke      (20 sentences per split, runs/smoke/llm/)

Slow (about 3 s per sentence on one GPU, ~5 hours for dev and test). Each answer is
saved as soon as it arrives; rerunning the command skips sentences already done.
--workers > 1 only helps if the Ollama server runs requests in parallel (OLLAMA_NUM_PARALLEL).

Outputs in runs/llm/<model>/ :
  raw_{dev,test}.jsonl        the model's answer per sentence (the resume cache)
  config.json                 model, options, prompt
  thresholds.json             theta tuned on dev
  dev.jsonl, test.jsonl       predictions with theta, for evaluation/evaluate.py
  {dev,test}_untuned.jsonl    every technique the model named
"""
import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from evaluation.evaluate import write_predictions
from src.common import (RUNS, add_smoke_option, apply_thresholds, load_jsonl, load_split, matrix, save_json,
                        technique_pool, tune_thresholds)

DEFAULT_MODEL = "llama3.1:8b"
OPTIONS = {"temperature": 0, "num_ctx": 4096}
SPLITS = ("dev", "test")


def first_sentence(text):
    return re.split(r"(?<=\.)\s", text, maxsplit=1)[0]


def system_prompt(pool):
    techniques = json.load(open("data/techniques.json"))["techniques"]
    catalog = "\n".join(f"{t} - {techniques[t]['name']}: {first_sentence(techniques[t]['description'])}"
                        for t in pool)
    return f"""You are a cyber threat intelligence analyst. Given one sentence from a threat report, \
identify which of the MITRE ATT&CK techniques below the sentence describes an adversary performing.

Rules:
- Select a technique only if the sentence describes that behavior. Do not infer from general knowledge.
- Most sentences describe no technique. If none apply, return an empty list.
- A sentence may describe more than one technique.
- For each selected technique, give a confidence between 0 and 1.

Techniques:
{catalog}"""


def schema(pool):
    return {
        "type": "object",
        "properties": {
            "techniques": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"id": {"type": "string", "enum": list(pool)}, "confidence": {"type": "number"}},
                    "required": ["id", "confidence"],
                },
            }
        },
        "required": ["techniques"],
    }


def parse(content, pool):
    """({technique: confidence}, ok). Unknown ids are dropped, confidences clipped to [0, 1],
    a technique named twice keeps its highest confidence. ok=False when the answer is not valid JSON."""
    try:
        items = json.loads(content)["techniques"]
        scores = {}
        for item in items:
            if item["id"] in pool:
                c = min(1.0, max(0.0, float(item["confidence"])))
                scores[item["id"]] = max(c, scores.get(item["id"], 0.0))
        return scores, True
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return {}, False


def ask(client, model, system, fmt, pool, sentence):
    response = client.chat(model=model, messages=[{"role": "system", "content": system},
                                                  {"role": "user", "content": f"Sentence: {sentence}"}],
                           format=fmt, think=False, options=OPTIONS)
    return parse(response.message.content, pool)


def label_split(rows, raw_path, model, system, fmt, pool, workers):
    """Ask the model about every row not yet in raw_path, appending each answer as it arrives."""
    import ollama

    done = {(r["doc_id"], r["sent_idx"]) for r in load_jsonl(raw_path)} if raw_path.exists() else set()
    todo = [r for r in rows if (r["doc_id"], r["sent_idx"]) not in done]
    print(f"{raw_path.name}: {len(done)} done, {len(todo)} to go")
    if not todo:
        return
    client = ollama.Client()
    start, failures = time.time(), 0
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_path, "a") as f, ThreadPoolExecutor(workers) as pool_exec:
        futures = {pool_exec.submit(ask, client, model, system, fmt, pool, r["text"]): r for r in todo}
        for n, future in enumerate(as_completed(futures), 1):
            r = futures[future]
            scores, ok = future.result()
            failures += not ok
            f.write(json.dumps({"doc_id": r["doc_id"], "sent_idx": r["sent_idx"], "scores": scores, "ok": ok}) + "\n")
            f.flush()
            if n % 100 == 0 or n == len(todo):
                rate = (time.time() - start) / n
                print(f"  {n}/{len(todo)}  {rate:.2f} s/sentence  ~{rate * (len(todo) - n) / 60:.0f} min left"
                      f"  parse failures {failures}", flush=True)


def load_raw(raw_path):
    rows = load_jsonl(raw_path)
    return {(r["doc_id"], r["sent_idx"]): r["scores"] for r in rows}, sum(not r["ok"] for r in rows)


def out_dir(args):
    return RUNS / "smoke" / "llm" if args.smoke else RUNS / "llm" / args.model.replace(":", "-").replace("/", "-")


def predict(args):
    out = out_dir(args)
    pool = technique_pool("full", "test")  # all 40 in-scope techniques; the LLM is not trained
    system, fmt = system_prompt(pool), schema(pool)
    save_json(out / "config.json", {"model": "llm", "ollama_model": args.model, "options": OPTIONS,
                                    "limit": args.limit, "pool": pool, "system_prompt": system})

    raw, failures = {}, {}
    rows = {split: load_split(split, "full", args.limit) for split in SPLITS}  # every sentence
    for split in SPLITS:
        label_split(rows[split], out / f"raw_{split}.jsonl", args.model, system, fmt, pool, args.workers)
        raw[split], failures[split] = load_raw(out / f"raw_{split}.jsonl")

    S, G = matrix(rows["dev"], raw["dev"], pool)
    best = tune_thresholds(S, G, metric=args.metric)
    save_json(out / "thresholds.json", {"tuned_on": "dev", "metric": args.metric, "parse_failures": failures, **best})
    print(f"dev: theta={best['theta']:.2f}  micro F1 {best['micro_f1']:.3f}  macro F1 {best['macro_f1']:.3f}"
          f"  (P {best['micro_precision']:.3f} R {best['micro_recall']:.3f})  parse failures {failures}")

    for split in SPLITS:
        S, _ = matrix(rows[split], raw[split], pool)
        keep = apply_thresholds(S, None, best["theta"])
        for name, mask in ((split, keep), (f"{split}_untuned", np.isfinite(S))):
            write_predictions(out / f"{name}.jsonl", [
                {"doc_id": r["doc_id"], "sent_idx": r["sent_idx"],
                 "predicted": [pool[j] for j in np.flatnonzero(mask[i])],
                 "scores": raw[split][(r["doc_id"], r["sent_idx"])]}
                for i, r in enumerate(rows[split])])
    print(f"predictions -> {out}/dev.jsonl, {out}/test.jsonl (and *_untuned.jsonl)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("predict",))
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model tag")
    ap.add_argument("--workers", type=int, default=1, help="requests sent to Ollama at once")
    ap.add_argument("--limit", type=int, help="first N sentences per split (smoke runs; evaluate.py rejects them)")
    ap.add_argument("--metric", default="micro_f1", choices=("micro_f1", "macro_f1"), help="dev score that picks theta")
    add_smoke_option(ap)
    args = ap.parse_args()
    if args.smoke:
        args.limit = args.limit or 20
    predict(args)


if __name__ == "__main__":
    main()
