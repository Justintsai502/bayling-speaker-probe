# bayling-speaker-probe

Which layers of BayLing-Duplex carry speaker identity?

The answer decides where BayLing-MP should inject Mimi features, instead of adding
them at the input embedding and disturbing all 40 layers.

## Design

**Control content, vary only the voice.** Qwen3-TTS clones N voices, and every voice reads
the same 40 Harvard sentences. That gives a complete speaker x sentence table, so
a probe cannot identify the speaker from the words.

**Feed audio the way BayLing actually runs.** The layout was read from `bayling_duplex/duplex.py`
(`stream_audio_tokens`) and the released `modeling_chatglm.py`:

- There is no system prompt, and position ids run 0..L-1.
- Every 0.8 s block is `[user audio x10][text x5][assistant audio x10]`.
- The audio token id is `<|audio_0|>` (152353) plus the GLM speech-tokenizer index.
- The state tokens are `[SILENCE]` 168736, `[PAD]` 168737, `[EPAD]` 168738 and `<|assistant|>` 151337.
- `output_hidden_states=True` returns 41 tensors: index 0 is the embedding output and index k is the output of block k.

The user channel is 1.6 s of silence, then the trimmed sentence, then zero padding to a block boundary.
Speech starts on a block boundary, so every user token is labeled speech or silence from known offsets.
Hidden states are **mean-pooled over the speech tokens only**.

The assistant channel has three `--context` modes:

| mode | assistant channel | why |
|---|---|---|
| `silence` (default) | `[SILENCE]` x5, plus speech tokens of digital silence | Identical for every clip, so only the user audio differs. This matches what BayLing does while listening. One forward pass. |
| `greedy` | The model decodes its own tokens (argmax) | Exactly like inference. Its reactions may differ per clip. |
| `user_only` | none | Out of distribution. Ablation only. |

**Two ways to read each layer:**

| metric | how | reads as |
|---|---|---|
| `spk_acc` | Ridge linear probe for speaker, cross-validated over **sentences** | Speaker information that is decodable on unseen text |
| `content_acc` | The same probe for sentence id, cross-validated over **speakers** | The contrast curve |
| `spk_index` | mean cos(same speaker, different sentence) − mean cos(different speaker, same sentence), on z-scored features | Above 0 means the layer groups clips by voice rather than by words. Needs no training. |

**Baselines** sit on the same plot:
- ECAPA is the ceiling.
- Mimi CB1 is the semantic codebook.
- Mimi CB2–8 is what BayLing-MP would inject.
- Mimi CB1–8 is what Moshi reads.
- Chance is 1/N.

## Pipeline

| stage | script | runs on | output |
|---|---|---|---|
| 1 | `spkprobe/tts_clone.py` | GPU, TTS env | `work/tts/wavs/`, `manifest.jsonl` |
| 2 | `spkprobe/check_tts.py` | GPU | `qc.tsv`, `manifest.kept.jsonl` |
| 3 | `spkprobe/extract_bayling.py` | GPU (9B, bf16 ≈ 19 GB) | `work/features/bayling_<context>.npz` |
| 4 | `spkprobe/extract_baselines.py` | GPU | `work/features/baselines.npz` |
| 5 | `spkprobe/analyze.py` | CPU, numpy only | `work/results/metrics.csv`, `summary.md`, `curves.png` |

Stage 2 filters out clips that would break the control:
- The speaking rate is out of range.
- The clip is clipped.
- ECAPA similarity to the voice prompt is too low.
- With `--asr`, Whisper WER is above 0.2.

It also prints whether ECAPA can tell the cloned voices apart. If it cannot, the voices collapsed and the probe has nothing to find.

## Everything stays inside the project

| folder | contents |
|---|---|
| `work/` | TTS audio, manifests, features, results, logs (`work/logs/`) |
| `cache/` | Hugging Face / torch / speechbrain downloads (Qwen3-TTS, Whisper, ECAPA, Mimi) |
| `models/` | BayLing-Duplex checkpoint and GLM speech tokenizer |

Every script sets `HF_HOME`, `TORCH_HOME`, `XDG_CACHE_HOME` and `MPLCONFIGDIR` to `cache/` on start, overriding values from `~/.bashrc`. Default paths are resolved from the project root, not the current directory. All three folders are git-ignored.

## Setup (server)

```bash
# models for BayLing (from its README)
hf download BayLing-Models/BayLing-Duplex --local-dir models/bayling_duplex_model
hf download zai-org/glm-4-voice-tokenizer --local-dir models/speech_tokenizer

# probe env
git clone https://github.com/BayLing-Models/BayLing-Duplex && pip install -e BayLing-Duplex
pip install -r requirements-probe.txt

# TTS env (separate is safer: qwen-tts pins transformers==4.57.3)
pip install -r requirements-tts.txt
```

BayLing's released `config.json` also reports transformers 4.57.3, so a single environment may work. This has not been tested.

## Run

Put voice prompts in `voices/` (see `voices/README.md`), or reuse a manifest from `../voice-clone-prompts/build_prompts.py`.

```bash
# smoke test: 2 speakers x 3 sentences
SMOKE=1 VOICES=voices TTS_PY=~/envs/tts/bin/python PROBE_PY=~/envs/probe/bin/python bash scripts/run_server.sh

# full run
VOICES=voices TTS_PY=~/envs/tts/bin/python PROBE_PY=~/envs/probe/bin/python bash scripts/run_server.sh
```

Each stage also runs on its own; every script has `--help`.

## Offline check (laptop, numpy only)

```bash
python -m unittest discover -s tests -v
python spkprobe/tts_clone.py --voices voices --dry-run      # plan the table without a model
python spkprobe/fake_features.py --out work/fake             # features with a planted peak at layer 8
python spkprobe/analyze.py --bayling work/fake/bayling_fake.npz --baselines work/fake/baselines.npz --out work/fake/results
```

The tests confirm four things:
- The analysis recovers a planted speaker peak.
- The speaker probe stays near chance when features only encode the sentence, so content does not leak.
- The interleaving puts user tokens at the right positions.
- Speech starts right after the lead silence.

## Reading the result

| curve | meaning | injection layer |
|---|---|---|
| clear peak in the middle | acoustic cues are processed there | the peak |
| highest at layer 0, then decays | the cue lives only in the token and washes out | early |
| near chance everywhere | GLM tokens carry no timbre | probe cannot choose; run short injection trials at a few layers |

A probe shows that information is decodable, not that the model uses it. Only injection experiments show causality.

## Assumptions to keep in mind

- **Silence in the assistant channel.** The `silence` context assumes BayLing's listening state looks like `[SILENCE]` text plus tokenized digital silence. Compare it against `greedy` to check.
- **TTS voices.** Cloned voices may be cleaner and more separable than real speakers. Re-check the conclusion on real recordings (e.g. VCTK shared paragraphs) before relying on it.
- **Pooling.** Mean pooling over whole utterances measures utterance-level identity. For frame-level questions, rerun stage 3 with `--save-frames`.
