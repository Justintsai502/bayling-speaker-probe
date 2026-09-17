#!/usr/bin/env bash
# Full pipeline on a GPU server. Run from the project root:
#   VOICES=voices bash scripts/run_server.sh
# or with a build_prompts.py manifest:
#   PROMPT_MANIFEST=/path/to/manifest.jsonl bash scripts/run_server.sh
#
# Optional env:
#   TTS_PY / PROBE_PY   python executables of the two environments (default: python)
#   BAYLING_REPO        BayLing-Duplex checkout, if not pip-installed
#   MODELS              directory with bayling_duplex_model/ and speech_tokenizer/ (default: models)
#   SMOKE=1             2 speakers x 3 sentences, first 6 utterances only
set -euo pipefail

TTS_PY=${TTS_PY:-python}
PROBE_PY=${PROBE_PY:-python}
MODELS=${MODELS:-models}
WORK=${WORK:-work}

voice_args=()
[[ -n "${VOICES:-}" ]] && voice_args+=(--voices "$VOICES")
[[ -n "${PROMPT_MANIFEST:-}" ]] && voice_args+=(--prompt-manifest "$PROMPT_MANIFEST")
if [[ ${#voice_args[@]} -eq 0 ]]; then
  echo "set VOICES and/or PROMPT_MANIFEST" >&2
  exit 1
fi

smoke_tts=() smoke_ext=()
if [[ "${SMOKE:-0}" == "1" ]]; then
  smoke_tts=(--max-speakers 2 --max-sentences 3)
  smoke_ext=(--limit 6)
fi

repo_args=()
[[ -n "${BAYLING_REPO:-}" ]] && repo_args+=(--bayling-repo "$BAYLING_REPO")

echo "== 1/5 TTS voice cloning"
$TTS_PY spkprobe/tts_clone.py ${voice_args[@]+"${voice_args[@]}"} ${smoke_tts[@]+"${smoke_tts[@]}"} --out "$WORK/tts"

echo "== 2/5 TTS quality check"
$PROBE_PY spkprobe/check_tts.py --tts-dir "$WORK/tts" --asr

echo "== 3/5 BayLing hidden states (silence context, then greedy ablation)"
for ctx in silence greedy; do
  $PROBE_PY spkprobe/extract_bayling.py ${repo_args[@]+"${repo_args[@]}"} ${smoke_ext[@]+"${smoke_ext[@]}"} \
    --manifest "$WORK/tts/manifest.kept.jsonl" \
    --model-path "$MODELS/bayling_duplex_model" \
    --speech-tokenizer-path "$MODELS/speech_tokenizer" \
    --context "$ctx" --out "$WORK/features/bayling_$ctx.npz"
done

echo "== 4/5 Baselines (ECAPA, Mimi)"
$PROBE_PY spkprobe/extract_baselines.py ${smoke_ext[@]+"${smoke_ext[@]}"} \
  --manifest "$WORK/tts/manifest.kept.jsonl" --out "$WORK/features/baselines.npz"

echo "== 5/5 Analysis"
$PROBE_PY spkprobe/analyze.py \
  --bayling "$WORK/features/bayling_silence.npz" "$WORK/features/bayling_greedy.npz" \
  --baselines "$WORK/features/baselines.npz" --out "$WORK/results"

echo "done: $WORK/results/summary.md"
