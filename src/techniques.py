"""Build the technique corpus: one text representation per ATT&CK parent technique.

Reads an ATT&CK Enterprise STIX bundle and writes data/techniques.json:

  {"attack_version": "13.1",
   "techniques": {"T1547": {"name": ..., "tactics": [...], "description": ..., "text": ...}, ...}}

`text` is what every model encodes: "<name>: <first paragraph of description>".
The full cleaned description is kept too, in case you later test longer inputs.

Usage:
  python techniques.py --stix raw/tram/enterprise-attack.json --labels data/label_map.json
"""
import argparse, json, re
from pathlib import Path

CITATION = re.compile(r"\(Citation:[^)]*\)")    # (Citation: Microsoft Run Key)
MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")  # [text](https://...) -> text
HTML_TAG = re.compile(r"</?[a-zA-Z][^>]*>")     # <code>, </code>, ...


def clean(text):
    text = CITATION.sub("", text)
    text = MD_LINK.sub(r"\1", text)
    text = HTML_TAG.sub("", text)
    paragraphs = [" ".join(p.split()) for p in text.split("\n\n")]
    return [p for p in paragraphs if p]


def attack_id(obj):
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack":
            return ref.get("external_id")
    return None


def build(stix_path):
    bundle = json.load(open(stix_path))
    version = next((o.get("x_mitre_version") for o in bundle["objects"]
                    if o["type"] == "x-mitre-collection"), "unknown")
    techniques = {}
    for o in bundle["objects"]:
        if o["type"] != "attack-pattern":
            continue
        if o.get("revoked") or o.get("x_mitre_deprecated") or o.get("x_mitre_is_subtechnique"):
            continue
        tid = attack_id(o)
        if not tid:
            continue
        paragraphs = clean(o.get("description", ""))
        techniques[tid] = {
            "name": o["name"],
            "tactics": sorted(p["phase_name"] for p in o.get("kill_chain_phases", [])
                              if p.get("kill_chain_name") == "mitre-attack"),
            "description": "\n\n".join(paragraphs),
            "text": f"{o['name']}: {paragraphs[0] if paragraphs else ''}",
        }
    return version, dict(sorted(techniques.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stix", type=Path, default=Path("data/raw/tram/attack/enterprise-attack.json"))
    ap.add_argument("--labels", type=Path, default=Path("data/label_map.json"),
                    help="label_map.json from build_dataset.py, used to check coverage")
    ap.add_argument("--out", type=Path, default=Path("data/techniques.json"))
    a = ap.parse_args()

    version, techniques = build(a.stix)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"attack_version": version, "source": str(a.stix), "techniques": techniques},
              open(a.out, "w"), indent=1)
    print(f"ATT&CK v{version}: {len(techniques)} parent techniques -> {a.out}")

    # Every TRAM label must have a description, or that technique can't be retrieved.
    if a.labels.exists():
        labels = json.load(open(a.labels))
        missing = sorted(set(labels) - set(techniques))
        print(f"TRAM labels covered: {len(labels) - len(missing)}/{len(labels)}")
        if missing:
            raise SystemExit(f"Missing descriptions for: {missing}")


if __name__ == "__main__":
    main()