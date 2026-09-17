#!/usr/bin/env python3
"""
Stage 1 - synthesize the speaker x sentence table with Qwen3-TTS voice cloning.

Every speaker reads every sentence (same text, different voice), so a probe
cannot use content to guess the speaker.

Voice prompts, either:
  --voices DIR               DIR/<speaker>.wav + DIR/<speaker>.txt (exact transcript)
                             a missing .txt falls back to x-vector-only cloning
  --prompt-manifest FILE     manifest.jsonl from voice-clone-prompts/build_prompts.py
                             (first prompt per speaker is used)

Output (--out):
  wavs/<speaker>/<sentence_id>_t<take>.wav
  manifest.jsonl             one row per utterance

Hard deps: numpy
Model deps (lazy): torch, soundfile, qwen-tts  (pip install -U qwen-tts)
"""

import argparse
import random
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spkprobe.common import read_jsonl, read_sentences, utt_id, write_jsonl  # noqa: E402


# ---------------------------------------------------------------- prompts

def collect_prompts(voices=None, prompt_manifest=None):
    """Return [{speaker, ref_wav, ref_text or None}] sorted by speaker."""
    prompts = {}
    if voices:
        vdir = Path(voices)
        for wav in sorted(vdir.glob("*.wav")):
            txt = wav.with_suffix(".txt")
            text = txt.read_text(encoding="utf-8").strip() if txt.exists() else None
            prompts[wav.stem] = {"speaker": wav.stem, "ref_wav": str(wav), "ref_text": text or None}
    if prompt_manifest:
        root = Path(prompt_manifest).parent
        for r in read_jsonl(prompt_manifest):
            spk = str(r["speaker"])
            if spk in prompts:
                continue
            prompts[spk] = {
                "speaker": spk,
                "ref_wav": str(root / r["prompt_wav"]),
                "ref_text": r.get("prompt_text") or None,
            }
    return [prompts[k] for k in sorted(prompts)]


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voices", type=Path, default=None)
    ap.add_argument("--prompt-manifest", type=Path, default=None)
    ap.add_argument("--sentences", type=Path, default=Path("data/harvard_sentences.txt"))
    ap.add_argument("--out", type=Path, default=Path("work/tts"))
    ap.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    ap.add_argument("--language", default="English")
    ap.add_argument("--max-speakers", type=int, default=0, help="0 = all")
    ap.add_argument("--max-sentences", type=int, default=0, help="0 = all")
    ap.add_argument("--takes", type=int, default=1, help="takes per speaker x sentence")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--x-vector-only", action="store_true",
                    help="clone from speaker embedding only, even when a transcript exists")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--attn", default="flash_attention_2",
                    help="attn_implementation; use sdpa if flash-attn is not installed")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and write the manifest without loading the model")
    cfg = ap.parse_args()

    if not cfg.voices and not cfg.prompt_manifest:
        raise SystemExit("give --voices and/or --prompt-manifest")

    prompts = collect_prompts(cfg.voices, cfg.prompt_manifest)
    if cfg.max_speakers:
        prompts = prompts[:cfg.max_speakers]
    sentences = read_sentences(cfg.sentences)
    if cfg.max_sentences:
        sentences = sentences[:cfg.max_sentences]
    if len(prompts) < 2:
        raise SystemExit(f"need at least 2 speakers, found {len(prompts)}")

    n_xvec = sum(1 for p in prompts if cfg.x_vector_only or not p["ref_text"])
    print(f"{len(prompts)} speakers x {len(sentences)} sentences x {cfg.takes} takes "
          f"= {len(prompts) * len(sentences) * cfg.takes} utterances")
    if n_xvec:
        print(f"note: {n_xvec} speaker(s) use x-vector-only cloning (no transcript or --x-vector-only)")

    # plan every utterance first so the manifest is identical with or without the model
    rows = []
    for p in prompts:
        for sid, text in sentences:
            for take in range(cfg.takes):
                rows.append({
                    "utt_id": utt_id(p["speaker"], sid, take),
                    "speaker": p["speaker"],
                    "sentence_id": sid,
                    "take": take,
                    "text": text,
                    "wav": f"wavs/{p['speaker']}/{sid}_t{take}.wav",
                    "ref_wav": p["ref_wav"],
                    "ref_text": p["ref_text"],
                    "x_vector_only": bool(cfg.x_vector_only or not p["ref_text"]),
                })

    if cfg.dry_run:
        write_jsonl(cfg.out / "manifest.planned.jsonl", rows)
        print(f"dry run: wrote plan to {cfg.out / 'manifest.planned.jsonl'}")
        return

    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    tts = Qwen3TTSModel.from_pretrained(cfg.model, device_map=cfg.device,
                                        dtype=torch.bfloat16, attn_implementation=cfg.attn)
    gen_kwargs = dict(max_new_tokens=2048, do_sample=True, top_k=cfg.top_k, top_p=1.0,
                      temperature=cfg.temperature, repetition_penalty=1.05,
                      subtalker_dosample=True, subtalker_top_k=cfg.top_k, subtalker_top_p=1.0,
                      subtalker_temperature=cfg.temperature)

    done = []
    for p in prompts:
        todo = [r for r in rows if r["speaker"] == p["speaker"]
                and (cfg.overwrite or not (cfg.out / r["wav"]).exists())]
        mine = [r for r in rows if r["speaker"] == p["speaker"]]
        if todo:
            xvec = bool(cfg.x_vector_only or not p["ref_text"])
            # build the prompt once per speaker; every sentence reuses it
            prompt_items = tts.create_voice_clone_prompt(
                ref_audio=p["ref_wav"], ref_text=None if xvec else p["ref_text"],
                x_vector_only_mode=xvec)
            for i in range(0, len(todo), cfg.batch_size):
                batch = todo[i:i + cfg.batch_size]
                # crc32, not hash(): str hashes change between Python processes
                seed = cfg.seed + zlib.crc32(f"{p['speaker']}|{i}".encode())
                random.seed(seed)
                np.random.seed(seed % (2 ** 32))
                torch.manual_seed(seed)
                wavs, sr = tts.generate_voice_clone(
                    text=[r["text"] for r in batch], language=cfg.language,
                    voice_clone_prompt=prompt_items * len(batch), **gen_kwargs)
                for r, w in zip(batch, wavs):
                    path = cfg.out / r["wav"]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(str(path), np.asarray(w, dtype=np.float32), sr)
                print(f"[{p['speaker']}] {min(i + cfg.batch_size, len(todo))}/{len(todo)}")

        for r in mine:
            path = cfg.out / r["wav"]
            if path.exists():
                info = sf.info(str(path))
                done.append(dict(r, sr=info.samplerate, dur=round(info.duration, 3)))

    write_jsonl(cfg.out / "manifest.jsonl", done)
    print(f"wrote {len(done)} utterances to {cfg.out / 'manifest.jsonl'}")


if __name__ == "__main__":
    sys.exit(main())
