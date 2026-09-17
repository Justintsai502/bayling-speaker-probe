"""
Shared helpers: manifests, audio I/O, silence trimming, and the speaker x sentence grid.

Hard deps: numpy
Audio deps (lazy): soundfile, and soxr or scipy for resampling
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------- manifests

def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_sentences(path):
    """One sentence per line; blank lines and '#' comments are skipped.

    Returns [(sentence_id, text)] with ids s000, s001, ... in file order.
    """
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        t = line.strip()
        if not t or t.startswith("#"):
            continue
        out.append((f"s{len(out):03d}", t))
    return out


def utt_id(speaker, sentence_id, take=0):
    return f"{speaker}__{sentence_id}__t{take}"


# ---------------------------------------------------------------- audio

def load_audio(path, sr=None):
    """Load mono float32 audio; resample to `sr` when given."""
    import soundfile as sf

    x, sr_in = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr is not None and sr != sr_in:
        x = resample(x, sr_in, sr)
        sr_in = sr
    return x.astype(np.float32), sr_in


def resample(x, sr_in, sr_out):
    if sr_in == sr_out:
        return x
    try:
        import soxr
        return soxr.resample(x, sr_in, sr_out).astype(np.float32)
    except ImportError:
        pass
    from scipy.signal import resample_poly
    g = math.gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def frame_db(x, sr, win=0.020, hop=0.010):
    n, h = int(win * sr), int(hop * sr)
    if len(x) < n:
        return np.array([]), h
    frames = np.lib.stride_tricks.sliding_window_view(x, n)[::h]
    rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1))
    return 20.0 * np.log10(rms + 1e-12), h


def trim_silence(x, sr, range_db=40.0, floor_db=-50.0, keep_pad=0.0):
    """Cut leading/trailing silence. Returns (audio, start_sample, end_sample)."""
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    if peak <= 0:
        return x[:0], 0, 0
    db, hop = frame_db(x, sr)
    if db.size == 0:
        return x, 0, len(x)
    thr = max(20 * math.log10(peak) - range_db, floor_db)
    voiced = db > thr
    if not voiced.any():
        return x[:0], 0, 0
    first = int(np.argmax(voiced))
    last = len(voiced) - 1 - int(np.argmax(voiced[::-1]))
    pad = int(keep_pad * sr)
    s = max(0, first * hop - pad)
    e = min(len(x), (last + 1) * hop + int(0.020 * sr) + pad)
    return x[s:e], s, e


# ---------------------------------------------------------------- grid

def complete_grid(speakers, sentences):
    """Keep only the speakers and sentences that form a complete table.

    `speakers` and `sentences` are parallel per-utterance labels. Sentences
    missing for any kept speaker are dropped first (a bad TTS take removes one
    sentence, not a whole speaker); speakers that then still miss sentences
    are dropped. Returns a boolean mask over utterances plus the kept labels.
    """
    speakers = np.asarray(speakers)
    sentences = np.asarray(sentences)
    have = defaultdict(set)
    for s, t in zip(speakers, sentences):
        have[s].add(t)

    spk_set = sorted(have)
    sent_set = sorted(set(sentences))
    while True:
        sent_keep = [t for t in sent_set if all(t in have[s] for s in spk_set)]
        if len(sent_keep) >= 2 or len(spk_set) <= 2:
            break
        # too few shared sentences: drop the speaker with the fewest sentences
        spk_set.remove(min(spk_set, key=lambda s: len(have[s])))
    mask = np.isin(speakers, spk_set) & np.isin(sentences, sent_keep)
    return mask, spk_set, sent_keep
