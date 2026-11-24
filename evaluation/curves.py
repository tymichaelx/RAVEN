"""Training curves: was each model trained properly?

Reads the history.json that every training run writes (one entry per epoch) and, for
each run type and model, draws two panels with one line per seed:

  Loss    train loss and dev loss. Train loss falling while dev loss climbs = overfitting.
  Metric  the score early stopping watches (retriever: dev recall@5; verifier: dev micro F1),
          next to the same score on a fixed sample of training sentences. A wide, growing
          gap = overfitting. Both low = underfitting. Dev still rising at the last allowed
          epoch = undertrained (raise --epochs).

It also prints a verdict per seed and saves it next to the figure:
  runs/<run>/<model>/curves.png, curves.json   (runs/smoke/<model>/ with --smoke)

Older histories have no dev loss or train-sample scores (they were added later); those
panels then show what exists and the verdict says what is missing.

Notes on reading the verifier's loss panel: train loss is the running mean over the epoch
(dropout on); dev loss is measured after the epoch (dropout off), on dev pairs mined with the
same settings as the training pairs. Compare their trends, not their exact values.

Usage:
  python -m evaluation.curves                 # every run type and model found under runs/
  python -m evaluation.curves --run full --model verifier
  python -m evaluation.curves --smoke
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUNS = Path("runs")
MODELS = ("retriever", "verifier")

# Fixed roles, same in both panels: train is always blue, dev always orange.
TRAIN, DEV = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#2b2b29", "#6b6a63", "#e4e3dd"

RISE = 0.05       # dev loss this far above its minimum counts as rising
STILL_UP = 0.002  # dev metric gain in the last epoch that counts as still improving


def load_runs(folder):
    """[(seed name, history, config)] for every seed folder (or the folder itself) with a history."""
    found = []
    for hist in sorted(folder.glob("seed*/history.json")) or sorted(folder.glob("history.json")):
        config_path = hist.parent / "config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        found.append((hist.parent.name, json.loads(hist.read_text()), config))
    return found


def watched_metric(config, model):
    """The history key early stopping used: 'recall@5' or 'micro_f1'."""
    selected = config.get("selected_on", "recall@5" if model == "retriever" else "dev micro_f1")
    return selected.split()[-1]


def series(history, key):
    pairs = [(h["epoch"], h[key]) for h in history if key in h]
    return [e for e, _ in pairs], [v for _, v in pairs]


def diagnose(history, config, metric):
    """Verdict and supporting numbers for one seed."""
    epochs, dev = series(history, metric)
    best = max(range(len(dev)), key=lambda i: (dev[i], -i))  # first epoch with the top score
    cap, ran = config.get("epochs"), len(history)
    out = {"epochs_run": ran, "epoch_cap": cap, "best_epoch": epochs[best], f"best_dev_{metric}": dev[best],
           "notes": []}
    notes = out["notes"]

    still_rising = best == ran - 1 and ran > 1 and dev[-1] - dev[-2] > STILL_UP and cap == ran
    _, dev_loss = series(history, "dev_loss")
    _, train_loss = series(history, "loss")
    overfit_epoch = None
    if dev_loss:
        low = min(range(len(dev_loss)), key=lambda i: dev_loss[i])
        rise = (dev_loss[-1] - dev_loss[low]) / dev_loss[low]
        out.update({"dev_loss_min_epoch": low + 1, "dev_loss_rise_since_min": rise})
        if low < len(dev_loss) - 1 and rise > RISE and train_loss[-1] < train_loss[low]:
            overfit_epoch = low + 1
            notes.append(f"dev loss lowest at epoch {low + 1}, then up {rise:.0%} while train loss kept falling")
    else:
        notes.append("no dev loss in this history (trained before it was logged): retrain for the loss check")

    _, train_metric = series(history, f"train_{metric}")
    if train_metric:
        gap = train_metric[best] - dev[best]
        out.update({f"train_{metric}_at_best": train_metric[best], "gap_at_best": gap,
                    "gap_last": train_metric[-1] - dev[-1]})
        notes.append(f"train/dev {metric} at best epoch: {train_metric[best]:.3f} / {dev[best]:.3f} (gap {gap:+.3f})")
    else:
        notes.append("no train-sample scores in this history: retrain for the train/dev gap")

    if ran == 1:
        out["verdict"] = "one epoch only: nothing to judge"
    elif still_rising:
        out["verdict"] = f"undertrained: dev {metric} still rising at the epoch cap ({cap}); raise --epochs"
    elif overfit_epoch:
        out["verdict"] = (f"overfits after about epoch {overfit_epoch}; the saved checkpoint is epoch "
                          f"{epochs[best]}, so the model on disk is fine")
    elif best < ran - 1:
        out["verdict"] = (f"plateau: dev {metric} stopped improving after epoch {epochs[best]} (the saved "
                          "checkpoint); a flat score alone does not show overfitting, dev loss does")
    else:
        out["verdict"] = f"converged: best at epoch {epochs[best]}, no sign of overfitting yet"
    return out


def style(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel("epoch", color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.get_major_locator().set_params(integer=True)


def line(ax, history, key, color, label):
    x, y = series(history, key)
    if x:
        ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=5, label=label, alpha=0.9)
    return x, y


def end_label(ax, x, y, text):
    """Name a line at its last point (the legend repeats it)."""
    if x:
        ax.annotate(text, (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=9, color=INK)


def plot(runs, model, metric, path, title):
    fig, (loss_ax, metric_ax) = plt.subplots(1, 2, figsize=(11, 4), facecolor="#fcfcfb")
    for ax in (loss_ax, metric_ax):
        ax.set_facecolor("#fcfcfb")
    for n, (seed, history, _) in enumerate(runs):
        first = n == 0
        tx, ty = line(loss_ax, history, "loss", TRAIN, "train" if first else None)
        dx, dy = line(loss_ax, history, "dev_loss", DEV, "dev" if first else None)
        mtx, mty = line(metric_ax, history, f"train_{metric}", TRAIN, "train sample" if first else None)
        mdx, mdy = line(metric_ax, history, metric, DEV, "dev" if first else None)
        best = max(range(len(mdy)), key=lambda i: (mdy[i], -i))
        for ax in (loss_ax, metric_ax):
            ax.axvline(mdx[best], color=MUTED, linestyle="--", linewidth=1,
                       label="best epoch (saved)" if first and ax is metric_ax else None)
        if first:
            end_label(loss_ax, tx, ty, "train")
            end_label(loss_ax, dx, dy, "dev")
            end_label(metric_ax, mtx, mty, "train")
            end_label(metric_ax, mdx, mdy, "dev")

    style(loss_ax, "Loss", "loss")
    style(metric_ax, f"{metric} (the score early stopping watches)", metric)
    for ax in (loss_ax, metric_ax):
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    seeds = ", ".join(seed for seed, *_ in runs)
    fig.suptitle(f"{title}  ({seeds})", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def report(folder, model, title):
    runs = load_runs(folder)
    if not runs:
        return False
    metric = watched_metric(runs[0][2], model)
    verdicts = {seed: diagnose(history, config, metric) for seed, history, config in runs}
    plot(runs, model, metric, folder / "curves.png", title)
    (folder / "curves.json").write_text(json.dumps(verdicts, indent=2) + "\n")

    print(f"\n{title}  -> {folder / 'curves.png'}")
    for seed, v in verdicts.items():
        cap = f"/{v['epoch_cap']}" if v["epoch_cap"] else ""
        print(f"  {seed}: {v['epochs_run']}{cap} epochs, best epoch {v['best_epoch']} "
              f"(dev {metric} {v[f'best_dev_{metric}']:.3f})")
        print(f"    verdict: {v['verdict']}")
        for note in v["notes"]:
            print(f"    - {note}")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", choices=("full", "withheld"), help="only this run type (default: all found)")
    ap.add_argument("--model", choices=MODELS, help="only this model (default: both)")
    ap.add_argument("--smoke", action="store_true", help="read runs/smoke/ instead")
    args = ap.parse_args()

    models = [args.model] if args.model else MODELS
    if args.smoke:
        targets = [(RUNS / "smoke" / m, m, f"smoke {m}") for m in models]
    else:
        runs = [args.run] if args.run else ("full", "withheld")
        targets = [(RUNS / r / m, m, f"{r} run, {m}") for r in runs for m in models]
    if not any([report(folder, model, title) for folder, model, title in targets]):
        raise SystemExit("no history.json found: train a model first")


if __name__ == "__main__":
    main()
