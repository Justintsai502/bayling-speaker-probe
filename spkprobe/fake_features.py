#!/usr/bin/env python3
"""
Offline stand-in for stages 1-4: write feature files with a KNOWN answer so the
analysis can be checked on a laptop with numpy only.

Planted structure, per layer l of L:
  speaker signal  strength peaks at --peak-layer (Gaussian bump)
  content signal  strongest at layer 0, decays with depth
  noise           unit Gaussian

Output: <out>/bayling_fake.npz and <out>/baselines.npz in the real formats.

Hard deps: numpy
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spkprobe.common import utt_id  # noqa: E402


def make(n_spk=8, n_sent=20, n_layers=13, dim=64, peak_layer=8, spk_gain=1.2, sent_gain=1.5, seed=0):
    rng = np.random.default_rng(seed)
    spk_vec = rng.normal(size=(n_spk, dim))
    sent_vec = rng.normal(size=(n_sent, dim))
    layers = np.arange(n_layers)
    a = spk_gain * np.exp(-0.5 * ((layers - peak_layer) / 1.5) ** 2)
    b = sent_gain * np.exp(-layers / 4.0)

    feats, spk, sent, utts = [], [], [], []
    for s in range(n_spk):
        for t in range(n_sent):
            x = (a[:, None] * spk_vec[s] + b[:, None] * sent_vec[t]
                 + rng.normal(size=(n_layers, dim)))
            feats.append(x)
            spk.append(f"spk{s:02d}")
            sent.append(f"s{t:03d}")
            utts.append(utt_id(spk[-1], sent[-1]))
    feats = np.stack(feats).astype(np.float32)
    ecapa = np.stack([2.0 * spk_vec[int(s[3:])] + rng.normal(size=dim) for s in spk]).astype(np.float32)
    return feats, ecapa, np.array(utts), np.array(spk), np.array(sent)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("work/fake"))
    ap.add_argument("--peak-layer", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    cfg = ap.parse_args()

    feats, ecapa, utts, spk, sent = make(peak_layer=cfg.peak_layer, seed=cfg.seed)
    cfg.out.mkdir(parents=True, exist_ok=True)
    np.savez(cfg.out / "bayling_fake.npz", pooled=feats, utt_id=utts, speaker=spk, sentence_id=sent,
             n_speech_frames=np.full(len(utts), 30), n_user_tokens=np.full(len(utts), 60),
             meta=json.dumps({"context": "fake", "peak_layer": cfg.peak_layer}))
    np.savez(cfg.out / "baselines.npz", feat_ecapa_fake=ecapa, utt_id=utts, speaker=spk,
             sentence_id=sent, meta=json.dumps({"features": ["ecapa_fake"]}))
    print(f"wrote {cfg.out}/bayling_fake.npz {feats.shape} (planted speaker peak at layer {cfg.peak_layer})")


if __name__ == "__main__":
    sys.exit(main())
