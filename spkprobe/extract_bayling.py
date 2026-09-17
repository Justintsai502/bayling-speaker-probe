#!/usr/bin/env python3
"""
Stage 3 - run BayLing-Duplex over every utterance and keep per-layer hidden states
at the user-speech positions.

Sequence layout (read from bayling_duplex/duplex.py, stream_audio_tokens):
  no system prompt; position ids run 0..L-1; each 0.8 s block is
    [user audio x10][text x5][assistant audio x10]
  audio token id = <|audio_0|> + GLM speech-tokenizer index

Audio layout fed to the user channel:
  [lead silence: --lead-blocks x 0.8 s][trimmed speech][zero pad to a block boundary]
Speech therefore starts on a block boundary, and every user token is labeled
speech / silence from the known offsets (no VAD needed).

Assistant channels (--context):
  silence   text = [SILENCE] x5, assistant audio = tokenizer output for digital
            silence. Identical for every utterance, so the only thing that differs
            between speakers is the user audio. One forward pass. (default)
  greedy    let the model decode its own text/audio (temperature 0), exactly like
            inference. Its outputs may differ per clip, which adds a confound.
  user_only only the user tokens, no interleaving. Out of distribution; ablation.

Hidden states: ChatGLM returns num_layers + 1 tensors; index 0 is the embedding
output (the GLM token itself), index k the output of block k (the last one before
the final RMSNorm).

Output (--out, default work/features/bayling_<context>.npz):
  pooled          [N, L+1, 4096]  mean over speech frames
  utt_id, speaker, sentence_id, n_speech_frames, n_user_tokens
  meta            JSON string with the settings
  frames/<utt_id>.npy  [L+1, n_speech_frames, 4096] float16  (only with --save-frames)

Hard deps: numpy
Model deps (lazy): torch, transformers, soundfile, scipy, bayling_duplex
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spkprobe.common import PROJECT_ROOT, setup_project_cache, load_audio, read_jsonl, trim_silence  # noqa: E402

SR = 16000
FRAME_S = 0.08          # GLM speech tokenizer: 12.5 Hz
BLOCK_TOKENS = 10       # user tokens per block for the released 10:5:10 checkpoint


# ---------------------------------------------------------------- layout (pure)

def interleaved_user_positions(n_blocks, x=10, y=5, z=10):
    """Sequence positions of the user tokens in [user x][text y][audio z] blocks."""
    period = x + y + z
    return (np.arange(n_blocks)[:, None] * period + np.arange(x)[None, :]).reshape(-1)


def build_interleaved_ids(user_ids, text_ids, audio_ids, x=10, y=5, z=10):
    """Interleave per-block channels into one id list. Lengths must be n_blocks * ratio."""
    n_blocks = len(user_ids) // x
    assert len(user_ids) == n_blocks * x
    assert len(text_ids) == n_blocks * y and len(audio_ids) == n_blocks * z
    seq = []
    for b in range(n_blocks):
        seq.extend(user_ids[b * x:(b + 1) * x])
        seq.extend(text_ids[b * y:(b + 1) * y])
        seq.extend(audio_ids[b * z:(b + 1) * z])
    return seq


def speech_frame_mask(n_frames, lead_s, speech_s, frame_s=FRAME_S, min_overlap=0.5):
    """True for tokens whose 80 ms window overlaps the speech span by >= min_overlap."""
    start = np.arange(n_frames) * frame_s
    end = start + frame_s
    overlap = np.clip(np.minimum(end, lead_s + speech_s) - np.maximum(start, lead_s), 0, None)
    return overlap >= min_overlap * frame_s


def prepare_waveform(x, sr, lead_blocks, block_s=BLOCK_TOKENS * FRAME_S):
    """Trim, add lead silence, pad to a block boundary. Returns (wave, lead_s, speech_s, n_blocks)."""
    speech, _, _ = trim_silence(x, sr)
    lead = np.zeros(int(round(lead_blocks * block_s * sr)), np.float32)
    body = np.concatenate([lead, speech])
    block_n = int(round(block_s * sr))
    n_blocks = int(np.ceil(len(body) / block_n))
    wave = np.zeros(n_blocks * block_n, np.float32)
    wave[:len(body)] = body
    return wave, len(lead) / sr, len(speech) / sr, n_blocks


# ---------------------------------------------------------------- model side

def as_seq_hidden(h, seq_len):
    """Return [seq, hidden] from a ChatGLM hidden state of shape [1, s, h] or [s, 1, h]."""
    if h.shape[0] == 1 and h.shape[1] == seq_len:
        return h[0]
    if h.shape[1] == 1 and h.shape[0] == seq_len:
        return h[:, 0]
    raise ValueError(f"unexpected hidden-state shape {tuple(h.shape)} for seq_len={seq_len}")


def stack_positions(hidden_states, positions, seq_len):
    """hidden_states: tuple of L+1 tensors -> float32 numpy [L+1, len(positions), H]."""
    import torch

    idx = torch.as_tensor(positions, device=hidden_states[0].device)
    return torch.stack([as_seq_hidden(h, seq_len).index_select(0, idx).float()
                        for h in hidden_states]).cpu().numpy()


class HiddenExtractor:
    def __init__(self, bd, context, x, y, z):
        self.bd, self.context = bd, context
        self.x, self.y, self.z = x, y, z
        self._silence_audio = {}

    def silence_audio_ids(self, n_blocks):
        """Assistant-channel ids for digital silence, cached per length."""
        if n_blocks not in self._silence_audio:
            import torch
            zeros = torch.zeros(1, int(round(n_blocks * self.z * FRAME_S * SR)))
            toks, _ = self.bd.tokenize_audio((zeros, SR))
            toks = (list(toks) + [toks[-1]] * (n_blocks * self.z))[:n_blocks * self.z]
            self._silence_audio[n_blocks] = [t + self.bd.audio_token_start for t in toks]
        return self._silence_audio[n_blocks]

    def forward_full(self, ids):
        import torch

        bd = self.bd
        input_ids = torch.tensor([ids], device=bd.device)
        pos = torch.arange(len(ids), device=bd.device)[None]
        with torch.no_grad():
            out = bd.model(input_ids=input_ids, position_ids=pos, use_cache=False,
                           output_hidden_states=True, return_dict=True)
        return out.hidden_states

    def run(self, user_tokens):
        """user_tokens: speech-tokenizer indices, length n_blocks * x -> [L+1, n_user, H]."""
        bd, x, y, z = self.bd, self.x, self.y, self.z
        n_blocks = len(user_tokens) // x
        user_ids = [t + bd.audio_token_start for t in user_tokens]

        if self.context == "user_only":
            hs = self.forward_full(user_ids)
            return stack_positions(hs, np.arange(len(user_ids)), len(user_ids))

        if self.context == "silence":
            if bd.silence_token_id is None:
                raise SystemExit("tokenizer has no [SILENCE] token; use --context greedy")
            text_ids = [bd.silence_token_id] * (n_blocks * y)
            ids = build_interleaved_ids(user_ids, text_ids, self.silence_audio_ids(n_blocks), x, y, z)
            hs = self.forward_full(ids)
            return stack_positions(hs, interleaved_user_positions(n_blocks, x, y, z), len(ids))

        return self._run_greedy(user_ids, n_blocks)

    def _run_greedy(self, user_ids, n_blocks):
        """Mirror of BayLingDuplex.stream_audio_tokens with argmax decoding and no early stop."""
        import torch

        bd, x, y, z = self.bd, self.x, self.y, self.z
        past, cur, chunks = None, 0, []

        def step(ids, hidden=False):
            nonlocal past, cur
            inp = torch.tensor([ids], device=bd.device)
            pos = torch.arange(cur, cur + len(ids), device=bd.device)[None]
            out = bd.model(input_ids=inp, position_ids=pos, past_key_values=past, use_cache=True,
                           output_hidden_states=hidden, return_dict=True)
            past, cur = out.past_key_values, cur + len(ids)
            return out

        with torch.no_grad():
            for b in range(n_blocks):
                out = step(user_ids[b * x:(b + 1) * x], hidden=True)
                chunks.append(stack_positions(out.hidden_states, np.arange(x), x))
                logits = out.logits[:, -1, :]
                for _ in range(y):
                    tok = bd._sample_token(logits, temperature=0.0, top_p=1.0, mode="text")
                    logits = step([tok]).logits[:, -1, :]
                for _ in range(z):
                    tok = bd._sample_token(logits, temperature=0.0, top_p=1.0, mode="audio")
                    logits = step([tok]).logits[:, -1, :]
        return np.concatenate(chunks, axis=1)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=(PROJECT_ROOT / "work/tts/manifest.kept.jsonl"))
    ap.add_argument("--tts-dir", type=Path, default=None,
                    help="root for relative wav paths (default: manifest's directory)")
    ap.add_argument("--bayling-repo", type=Path, default=None,
                    help="path to a BayLing-Duplex checkout if it is not pip-installed")
    ap.add_argument("--model-path", type=Path, default=(PROJECT_ROOT / "models/bayling_duplex_model"))
    ap.add_argument("--speech-tokenizer-path", type=Path, default=(PROJECT_ROOT / "models/speech_tokenizer"))
    ap.add_argument("--interleave-ratio", default="10:5:10")
    ap.add_argument("--context", choices=["silence", "greedy", "user_only"], default="silence")
    ap.add_argument("--lead-blocks", type=int, default=2, help="silent blocks before speech")
    ap.add_argument("--min-overlap", type=float, default=0.5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--torch-dtype", default="bfloat16")
    ap.add_argument("--save-frames", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="first N utterances only (smoke test)")
    ap.add_argument("--out", type=Path, default=None)
    cfg = ap.parse_args()
    setup_project_cache()

    rows = read_jsonl(cfg.manifest)
    if cfg.limit:
        rows = rows[:cfg.limit]
    root = cfg.tts_dir or cfg.manifest.parent
    out = cfg.out or (PROJECT_ROOT / "work/features") / f"bayling_{cfg.context}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)

    if cfg.bayling_repo:
        sys.path.insert(0, str(cfg.bayling_repo.resolve()))
    import torch
    from bayling_duplex.duplex import BayLingDuplex

    bd = BayLingDuplex(model_path=str(cfg.model_path),
                       speech_tokenizer_path=str(cfg.speech_tokenizer_path),
                       decoder_path=None, interleave_ratio=cfg.interleave_ratio,
                       device=cfg.device, torch_dtype=cfg.torch_dtype)
    ext = HiddenExtractor(bd, cfg.context, bd.x_ratio, bd.y_ratio, bd.z_ratio)
    block_s = bd.x_ratio * FRAME_S

    pooled, meta_rows = [], []
    frames_dir = out.parent / f"frames_{cfg.context}"
    for i, r in enumerate(rows):
        x, _ = load_audio(root / r["wav"], sr=SR)
        wave, lead_s, speech_s, n_blocks = prepare_waveform(x, SR, cfg.lead_blocks, block_s)
        toks, _ = bd.tokenize_audio((torch.from_numpy(wave)[None], SR))
        n_tok = (len(toks) // bd.x_ratio) * bd.x_ratio
        if n_tok < n_blocks * bd.x_ratio:
            print(f"warn {r['utt_id']}: tokenizer gave {len(toks)} tokens for {n_blocks} blocks")
        toks = toks[:n_tok]

        h = ext.run(toks)                                     # [L+1, n_tok, H]
        mask = speech_frame_mask(n_tok, lead_s, speech_s, min_overlap=cfg.min_overlap)
        if not mask.any():
            print(f"skip {r['utt_id']}: no speech frames")
            continue
        pooled.append(h[:, mask].mean(axis=1))
        meta_rows.append((r["utt_id"], r["speaker"], r["sentence_id"], int(mask.sum()), n_tok))
        if cfg.save_frames:
            frames_dir.mkdir(parents=True, exist_ok=True)
            np.save(frames_dir / f"{r['utt_id']}.npy", h[:, mask].astype(np.float16))
        if (i + 1) % 20 == 0 or i + 1 == len(rows):
            print(f"{i + 1}/{len(rows)}")

    pooled = np.stack(pooled)
    store = pooled.astype(np.float16)
    if not np.isfinite(store).all():
        print("float16 overflow in hidden states; storing float32")
        store = pooled.astype(np.float32)
    utt, spk, sent, n_sp, n_us = map(np.array, zip(*meta_rows))
    meta = {"context": cfg.context, "model_path": str(cfg.model_path),
            "interleave_ratio": bd.interleave_ratio, "lead_blocks": cfg.lead_blocks,
            "min_overlap": cfg.min_overlap, "n_hidden_states": int(pooled.shape[1]),
            "layer_index_note": "0 = embedding output, k = output of transformer block k"}
    np.savez(out, pooled=store, utt_id=utt, speaker=spk, sentence_id=sent,
             n_speech_frames=n_sp, n_user_tokens=n_us, meta=json.dumps(meta))
    print(f"wrote {out}: pooled {tuple(store.shape)}")


if __name__ == "__main__":
    sys.exit(main())
