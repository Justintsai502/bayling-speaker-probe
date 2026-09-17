#!/usr/bin/env python3
"""
Stage 2 - quality-check the synthesized table before probing.

The probe only means something if (a) every clip really says its sentence and
(b) clips of the same voice sound like that voice. Checks per utterance:

  speaking rate    words per second after trimming silence
  clipping         fraction of samples at |x| >= 0.999
  ref similarity   ECAPA cosine between the clip and its voice prompt
  WER (optional)   Whisper transcript vs the sentence  (--asr)

Also reports whether ECAPA can tell the cloned voices apart (nearest speaker
centroid, leave-one-out). If it cannot, the TTS voices collapsed and the probe
has nothing to find.

Output (next to the manifest):
  qc.tsv                   every check for every utterance
  manifest.kept.jsonl      rows that passed

Hard deps: numpy
Model deps (lazy): torch, soundfile, speechbrain; transformers for --asr
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spkprobe.common import PROJECT_ROOT, setup_project_cache, load_audio, read_jsonl, trim_silence, write_jsonl  # noqa: E402

WORD_RE = re.compile(r"[a-z']+")


# ---------------------------------------------------------------- text

def normalize_words(text):
    return WORD_RE.findall(text.lower())


def wer(ref, hyp):
    r, h = normalize_words(ref), normalize_words(hyp)
    if not r:
        return 0.0 if not h else 1.0
    d = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return float(d[len(h)]) / len(r)


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def centroid_loo_accuracy(emb, labels):
    """Leave-one-out nearest-centroid speaker accuracy on L2-normalized embeddings."""
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    labels = np.asarray(labels)
    classes = sorted(set(labels))
    sums = {c: emb[labels == c].sum(0) for c in classes}
    counts = {c: int((labels == c).sum()) for c in classes}
    hit = 0
    for i, y in enumerate(labels):
        best, best_s = None, -np.inf
        for c in classes:
            s_, n = sums[c], counts[c]
            if c == y:
                s_, n = s_ - emb[i], n - 1
            if n == 0:
                continue
            s = cosine(emb[i], s_ / n)
            if s > best_s:
                best, best_s = c, s
        hit += best == y
    return hit / len(labels)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tts-dir", type=Path, default=(PROJECT_ROOT / "work/tts"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--min-wps", type=float, default=1.2)
    ap.add_argument("--max-wps", type=float, default=5.0)
    ap.add_argument("--max-clip", type=float, default=0.001)
    ap.add_argument("--min-ref-sim", type=float, default=0.35,
                    help="ECAPA cosine to the voice prompt; check the printed distribution before trusting it")
    ap.add_argument("--asr", action="store_true", help="also run Whisper and filter on WER")
    ap.add_argument("--asr-model", default="openai/whisper-large-v3")
    ap.add_argument("--max-wer", type=float, default=0.2)
    cfg = ap.parse_args()
    setup_project_cache()

    from spkprobe.encoders import ECAPA_SR, ecapa_embed, load_ecapa

    rows = read_jsonl(cfg.tts_dir / "manifest.jsonl")
    ecapa = load_ecapa(cfg.device)

    asr = None
    if cfg.asr:
        from transformers import pipeline
        asr = pipeline("automatic-speech-recognition", model=cfg.asr_model, device=cfg.device)

    ref_cache = {}
    embs, results = [], []
    for r in rows:
        x, _ = load_audio(cfg.tts_dir / r["wav"], sr=ECAPA_SR)
        speech, _, _ = trim_silence(x, ECAPA_SR)
        dur = len(speech) / ECAPA_SR
        n_words = len(normalize_words(r["text"]))
        wps = n_words / dur if dur > 0 else 0.0
        clip = float(np.mean(np.abs(x) >= 0.999)) if len(x) else 1.0

        e = ecapa_embed(ecapa, speech) if dur > 0.3 else np.zeros(192, np.float32)
        if r["ref_wav"] not in ref_cache:
            ref, _ = load_audio(r["ref_wav"], sr=ECAPA_SR)
            ref_cache[r["ref_wav"]] = ecapa_embed(ecapa, trim_silence(ref, ECAPA_SR)[0])
        sim = cosine(e, ref_cache[r["ref_wav"]])

        res = {"utt_id": r["utt_id"], "speaker": r["speaker"], "sentence_id": r["sentence_id"],
               "speech_dur": round(dur, 3), "wps": round(wps, 2), "clip": round(clip, 5),
               "ref_sim": round(sim, 4), "wer": "", "hyp": ""}
        why = []
        if not (cfg.min_wps <= wps <= cfg.max_wps):
            why.append("rate")
        if clip > cfg.max_clip:
            why.append("clip")
        if sim < cfg.min_ref_sim:
            why.append("ref_sim")
        if asr is not None:
            hyp = asr({"raw": speech, "sampling_rate": ECAPA_SR})["text"]
            w = wer(r["text"], hyp)
            res.update(wer=round(w, 3), hyp=hyp.strip())
            if w > cfg.max_wer:
                why.append("wer")
        res["reject"] = ",".join(why)
        results.append(res)
        embs.append(e)

    # ---- report
    embs = np.stack(embs)
    sims = np.array([r["ref_sim"] for r in results])
    print(f"ref_sim: mean {sims.mean():.3f}  p10 {np.percentile(sims, 10):.3f}  "
          f"min {sims.min():.3f}")
    by_spk = defaultdict(list)
    for r in results:
        by_spk[r["speaker"]].append(r["ref_sim"])
    for spk in sorted(by_spk):
        print(f"  {spk}: mean ref_sim {np.mean(by_spk[spk]):.3f}")
    labels = [r["speaker"] for r in results]
    print(f"ECAPA nearest-centroid accuracy across cloned voices: "
          f"{centroid_loo_accuracy(embs, labels):.3f} (chance {1 / len(by_spk):.3f})")

    reasons = defaultdict(int)
    for r in results:
        for w in filter(None, r["reject"].split(",")):
            reasons[w] += 1
    kept_ids = {r["utt_id"] for r in results if not r["reject"]}
    print(f"kept {len(kept_ids)}/{len(results)}"
          + (" | rejected: " + ", ".join(f"{k}={v}" for k, v in sorted(reasons.items())) if reasons else ""))

    cols = ["utt_id", "speaker", "sentence_id", "speech_dur", "wps", "clip", "ref_sim", "wer", "reject", "hyp"]
    with open(cfg.tts_dir / "qc.tsv", "w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for r in results:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")
    write_jsonl(cfg.tts_dir / "manifest.kept.jsonl", [r for r in rows if r["utt_id"] in kept_ids])


if __name__ == "__main__":
    sys.exit(main())
