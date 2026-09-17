# Voice prompts

Put one reference clip per speaker here:

```
voices/
  alice.wav     5-15 s, one speaker, clean, no music
  alice.txt     exact transcript of alice.wav
  bob.wav
  bob.txt
```

- The file stem is the speaker id.
- The transcript must match the audio word for word; Qwen3-TTS uses it for in-context cloning.
- A missing `.txt` falls back to x-vector-only cloning (speaker embedding only, lower similarity).
- More speakers make the probe more reliable: aim for 8-20.

A `manifest.jsonl` from `../voice-clone-prompts/build_prompts.py` also works: pass `--prompt-manifest`.
