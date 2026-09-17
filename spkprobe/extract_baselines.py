#!/usr/bin/env python3
"""
Stage 4 - reference features on the same utterances, so the BayLing curve can be
read against a floor and a ceiling.

  ecapa           192-d speaker embedding          (ceiling: built for speaker ID)
  mimi_semantic   Mimi CB1, mean over frames       (distilled from WavLM, content-heavy)
  mimi_acoustic   Mimi CB2-8, mean over frames     (what BayLing-MP would inject)
  mimi_all        Mimi CB1-8, mean over frames     (what Moshi reads)

All features use the trimmed speech only, the same span BayLing is pooled over.

Output: work/features/baselines.npz with feat_<name> [N, D] plus the label arrays.

Hard deps: numpy
Model deps (lazy): torch, soundfile, speechbrain, transformers
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spkprobe.common import load_audio, read_jsonl, resample, trim_silence  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=Path("work/tts/manifest.kept.jsonl"))
    ap.add_argument("--tts-dir", type=Path, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip", nargs="*", default=[], choices=["ecapa", "mimi"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("work/features/baselines.npz"))
    cfg = ap.parse_args()

    from spkprobe.encoders import ECAPA_SR, MIMI_SR, ecapa_embed, load_ecapa, load_mimi, mimi_features

    rows = read_jsonl(cfg.manifest)
    if cfg.limit:
        rows = rows[:cfg.limit]
    root = cfg.tts_dir or cfg.manifest.parent

    ecapa = None if "ecapa" in cfg.skip else load_ecapa(cfg.device)
    mimi = None if "mimi" in cfg.skip else load_mimi(cfg.device)

    feats = {}
    for i, r in enumerate(rows):
        x, sr = load_audio(root / r["wav"])
        speech, _, _ = trim_silence(x, sr)
        if ecapa is not None:
            feats.setdefault("ecapa", []).append(ecapa_embed(ecapa, resample(speech, sr, ECAPA_SR)))
        if mimi is not None:
            m = mimi_features(*mimi, resample(speech, sr, MIMI_SR))
            for k in ("semantic", "acoustic", "all"):
                feats.setdefault(f"mimi_{k}", []).append(m[k].mean(axis=0))
        if (i + 1) % 50 == 0 or i + 1 == len(rows):
            print(f"{i + 1}/{len(rows)}")

    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"feat_{k}": np.stack(v).astype(np.float32) for k, v in feats.items()}
    np.savez(cfg.out,
             utt_id=np.array([r["utt_id"] for r in rows]),
             speaker=np.array([r["speaker"] for r in rows]),
             sentence_id=np.array([r["sentence_id"] for r in rows]),
             meta=json.dumps({"features": sorted(feats)}),
             **arrays)
    print(f"wrote {cfg.out}: " + ", ".join(f"{k} {v.shape}" for k, v in arrays.items()))


if __name__ == "__main__":
    sys.exit(main())
