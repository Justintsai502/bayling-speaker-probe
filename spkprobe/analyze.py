#!/usr/bin/env python3
"""
Stage 5 - where in BayLing does speaker identity live?

For every hidden-state layer (and every baseline feature) it computes:

  spk_acc       linear probe: which speaker?  Cross-validated over SENTENCES, so
                test clips are sentences the probe never saw.
  content_acc   linear probe: which sentence? Cross-validated over SPEAKERS.
                The contrast: where content rises, speaker usually falls.
  s_spk         mean cosine, same speaker / different sentence
  s_txt         mean cosine, different speaker / same sentence
  spk_index     s_spk - s_txt   (> 0: the layer groups clips by voice, not by words)

Probe = ridge classifier (one-vs-rest on one-hot targets), features z-scored on the
training fold, ridge strength picked by an inner split that is also grouped.
Only the complete speaker x sentence table is used, and every feature set is
restricted to the same utterances.

Output (--out):
  metrics.csv    one row per (feature set, layer)
  summary.md     best layers, baselines, chance levels
  curves.png     if matplotlib is installed

Hard deps: numpy
Optional: matplotlib
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spkprobe.common import complete_grid  # noqa: E402

LAMBDAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0)   # relative to mean(diag(XX^T))


# ---------------------------------------------------------------- CV splits

def group_folds(groups, n_splits, seed=0):
    """Assign whole groups to folds. Returns a list of (train_idx, test_idx)."""
    groups = np.asarray(groups)
    uniq = np.array(sorted(set(groups.tolist())))
    n_splits = min(n_splits, len(uniq))
    if n_splits < 2:
        raise ValueError(f"need >= 2 groups for cross-validation, got {len(uniq)}")
    rng = np.random.default_rng(seed)
    order = uniq[rng.permutation(len(uniq))]
    fold_of = {g: i % n_splits for i, g in enumerate(order)}
    fold = np.array([fold_of[g] for g in groups])
    return [(np.where(fold != k)[0], np.where(fold == k)[0]) for k in range(n_splits)]


# ---------------------------------------------------------------- ridge probe

def _zscore(train, *others):
    mu = train.mean(0)
    sd = train.std(0) + 1e-6
    return [(a - mu) / sd for a in (train,) + others]


class DualRidge:
    """Ridge in the dual (n << d): one eigendecomposition serves every lambda."""

    def __init__(self, X, Y):
        self.X = X
        self.Ymean = Y.mean(0)
        K = X @ X.T
        self.scale = float(np.trace(K)) / len(K)
        s, U = np.linalg.eigh(K)
        self.s, self.U = np.clip(s, 0, None), U
        self.UtY = U.T @ (Y - self.Ymean)

    def predict(self, Xte, lam):
        alpha = self.U @ (self.UtY / (self.s + lam * self.scale)[:, None])
        return (Xte @ self.X.T) @ alpha + self.Ymean


def probe_accuracy(X, y, groups, n_splits=5, inner_splits=3, seed=0):
    """Grouped CV accuracy of a ridge classifier. Returns (mean, std)."""
    X = np.asarray(X, dtype=np.float64)
    classes, yi = np.unique(np.asarray(y), return_inverse=True)
    Y = np.eye(len(classes))[yi]
    groups = np.asarray(groups)

    accs = []
    for tr, te in group_folds(groups, n_splits, seed):
        Xtr, Xte = _zscore(X[tr], X[te])

        # pick lambda on grouped inner folds of the training part
        if len(set(groups[tr].tolist())) < 2:
            inner = []                     # one training group: no inner split, keep lambda = 1
        else:
            inner = group_folds(groups[tr], inner_splits, seed + 1)
        scores = np.zeros(len(LAMBDAS))
        scores[LAMBDAS.index(1.0)] = 1e-9
        for itr, ite in inner:
            a, b = _zscore(X[tr][itr], X[tr][ite])
            model = DualRidge(a, Y[tr][itr])
            for j, lam in enumerate(LAMBDAS):
                scores[j] += np.mean(model.predict(b, lam).argmax(1) == yi[tr][ite])
        lam = LAMBDAS[int(np.argmax(scores))]

        pred = DualRidge(Xtr, Y[tr]).predict(Xte, lam).argmax(1)
        accs.append(np.mean(pred == yi[te]))
    return float(np.mean(accs)), float(np.std(accs))


# ---------------------------------------------------------------- similarity

def similarity_index(X, speakers, sentences):
    """Returns (s_spk, s_txt, s_spk - s_txt) on globally z-scored, L2-normalized features."""
    X = np.asarray(X, dtype=np.float64)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    C = X @ X.T
    spk, sent = np.asarray(speakers), np.asarray(sentences)
    same_spk = spk[:, None] == spk[None, :]
    same_sent = sent[:, None] == sent[None, :]
    s_spk = float(C[same_spk & ~same_sent].mean())
    s_txt = float(C[~same_spk & same_sent].mean())
    return s_spk, s_txt, s_spk - s_txt


# ---------------------------------------------------------------- loading

def load_sets(bayling_paths, baseline_path):
    """Return {set_name: (features [N, L, D], labels)} aligned on shared utterances."""
    raw = {}
    for p in bayling_paths:
        d = np.load(p, allow_pickle=False)
        meta = json.loads(str(d["meta"]))
        name = f"bayling_{meta.get('context', Path(p).stem)}"
        raw[name] = (d["pooled"], d["utt_id"], d["speaker"], d["sentence_id"])
    if baseline_path:
        d = np.load(baseline_path, allow_pickle=False)
        for k in d.files:
            if k.startswith("feat_"):
                raw[k[5:]] = (d[k][:, None, :], d["utt_id"], d["speaker"], d["sentence_id"])
    if not raw:
        raise SystemExit("no feature files given")

    shared = set.intersection(*(set(v[1].tolist()) for v in raw.values()))
    ref = next(iter(raw.values()))
    keep = np.array([u in shared for u in ref[1]])
    spk, sent = ref[2][keep], ref[3][keep]
    grid, spk_set, sent_set = complete_grid(spk, sent)
    utts = ref[1][keep][grid]

    out = {}
    for name, (F, u, s, t) in raw.items():
        pos = {x: i for i, x in enumerate(u.tolist())}
        idx = np.array([pos[x] for x in utts])
        out[name] = F[idx].astype(np.float32)
    labels = {"utt_id": utts, "speaker": spk[grid], "sentence_id": sent[grid]}
    return out, labels, spk_set, sent_set


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bayling", type=Path, nargs="*", default=[Path("work/features/bayling_silence.npz")])
    ap.add_argument("--baselines", type=Path, default=Path("work/features/baselines.npz"))
    ap.add_argument("--out", type=Path, default=Path("work/results"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--no-content", action="store_true", help="skip the sentence probe")
    ap.add_argument("--seed", type=int, default=0)
    cfg = ap.parse_args()

    bayling = [p for p in cfg.bayling if p.exists()]
    baselines = cfg.baselines if cfg.baselines and cfg.baselines.exists() else None
    for p in cfg.bayling:
        if not p.exists():
            print(f"missing, skipped: {p}")
    sets, labels, spk_set, sent_set = load_sets(bayling, baselines)
    spk, sent = labels["speaker"], labels["sentence_id"]
    n = len(spk)
    print(f"{n} utterances | {len(spk_set)} speakers x {len(sent_set)} sentences")

    rows = []
    for name, F in sets.items():
        for layer in range(F.shape[1]):
            X = F[:, layer]
            acc, acc_sd = probe_accuracy(X, spk, sent, cfg.folds, seed=cfg.seed)
            c_acc = c_sd = float("nan")
            if not cfg.no_content and len(spk_set) >= 2:
                c_acc, c_sd = probe_accuracy(X, sent, spk, cfg.folds, seed=cfg.seed)
            s_spk, s_txt, idx = similarity_index(X, spk, sent)
            rows.append({"set": name, "layer": layer if F.shape[1] > 1 else "",
                         "spk_acc": acc, "spk_acc_std": acc_sd,
                         "content_acc": c_acc, "content_acc_std": c_sd,
                         "s_spk": s_spk, "s_txt": s_txt, "spk_index": idx})
        last = rows[-1]
        print(f"{name}: done ({F.shape[1]} layer(s)); last spk_acc {last['spk_acc']:.3f}")

    cfg.out.mkdir(parents=True, exist_ok=True)
    with open(cfg.out / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

    write_summary(cfg.out / "summary.md", rows, len(spk_set), len(sent_set), n)
    plot_curves(cfg.out / "curves.png", rows, len(spk_set), len(sent_set))
    print(f"wrote {cfg.out}/metrics.csv, summary.md")


def write_summary(path, rows, n_spk, n_sent, n):
    lines = [f"# Speaker probe summary", "",
             f"- {n} utterances: {n_spk} speakers x {n_sent} sentences",
             f"- chance: speaker {1 / n_spk:.3f}, sentence {1 / n_sent:.3f}", ""]
    layered = sorted({r["set"] for r in rows if r["layer"] != ""})
    for name in layered:
        rs = [r for r in rows if r["set"] == name]
        by_acc = sorted(rs, key=lambda r: -r["spk_acc"])[:3]
        by_idx = sorted(rs, key=lambda r: -r["spk_index"])[:3]
        lines += [f"## {name}", "",
                  "| rank | layer (probe) | spk_acc | layer (index) | spk_index |",
                  "|---|---|---|---|---|"]
        for k, (a, b) in enumerate(zip(by_acc, by_idx), 1):
            lines.append(f"| {k} | {a['layer']} | {a['spk_acc']:.3f} | {b['layer']} | {b['spk_index']:.3f} |")
        l0 = rs[0]
        lines += ["", f"- layer 0 (GLM token embedding): spk_acc {l0['spk_acc']:.3f}, "
                      f"content_acc {l0['content_acc']:.3f}", ""]
    base = [r for r in rows if r["layer"] == ""]
    if base:
        lines += ["## Baselines", "", "| feature | spk_acc | content_acc | spk_index |", "|---|---|---|---|"]
        for r in base:
            lines.append(f"| {r['set']} | {r['spk_acc']:.3f} | {r['content_acc']:.3f} | {r['spk_index']:.3f} |")
        lines.append("")
    lines += ["## Reading the curve", "",
              "- clear peak in middle layers: the model processes acoustic cues there; inject at the peak",
              "- highest at layer 0, decaying: the cue lives only in the token and gets washed out; inject early",
              "- near chance everywhere: GLM tokens carry no timbre; pick the layer by short injection runs instead",
              "- a probe shows the information is decodable, not that the model uses it", ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def plot_curves(path, rows, n_spk, n_sent):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipped curves.png")
        return
    layered = sorted({r["set"] for r in rows if r["layer"] != ""})
    base = [r for r in rows if r["layer"] == ""]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name in layered:
        rs = [r for r in rows if r["set"] == name]
        L = [r["layer"] for r in rs]
        axes[0].plot(L, [r["spk_acc"] for r in rs], marker="o", ms=3, label=f"{name} speaker")
        if not np.isnan(rs[0]["content_acc"]):
            axes[0].plot(L, [r["content_acc"] for r in rs], ls="--", label=f"{name} sentence")
        axes[1].plot(L, [r["spk_index"] for r in rs], marker="o", ms=3, label=name)
    for r in base:
        axes[0].axhline(r["spk_acc"], lw=0.8, ls=":", color="gray")
        axes[0].annotate(r["set"], (0, r["spk_acc"]), fontsize=7, color="gray", va="bottom")
    axes[0].axhline(1 / n_spk, color="k", lw=0.8, label="speaker chance")
    axes[0].set(xlabel="layer (0 = embedding)", ylabel="probe accuracy", ylim=(0, 1.02),
                title="Linear probe (held-out sentences for speaker)")
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set(xlabel="layer (0 = embedding)", ylabel="s_spk - s_txt", title="Speaker similarity index")
    for a in axes:
        a.legend(fontsize=7)
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


if __name__ == "__main__":
    sys.exit(main())
