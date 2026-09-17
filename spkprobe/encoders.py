"""
Reference encoders used as probe baselines and for TTS quality control.

  ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb): 192-d speaker embedding, 16 kHz
  Mimi (kyutai/mimi via transformers):            8 RVQ codebooks at 12.5 Hz, 24 kHz

All heavy imports are lazy so this module imports with numpy alone.
"""

import numpy as np

ECAPA_SR = 16000
MIMI_SR = 24000


# ---------------------------------------------------------------- ECAPA

def load_ecapa(device="cuda", source="speechbrain/spkrec-ecapa-voxceleb", savedir=None):
    from spkprobe.common import CACHE_DIR

    # speechbrain otherwise writes pretrained_models/ into the current directory
    savedir = savedir or str(CACHE_DIR / "speechbrain" / source.split("/")[-1])
    try:
        from speechbrain.inference.speaker import EncoderClassifier  # speechbrain >= 1.0
    except ImportError:
        from speechbrain.pretrained import EncoderClassifier  # older speechbrain
    return EncoderClassifier.from_hparams(source=source, savedir=savedir,
                                          run_opts={"device": device})


def ecapa_embed(model, wav16k):
    """wav16k: 1-D float32 at 16 kHz -> (192,) float32."""
    import torch

    x = torch.from_numpy(np.asarray(wav16k, dtype=np.float32))[None]
    with torch.no_grad():
        e = model.encode_batch(x)
    return e.reshape(-1).float().cpu().numpy()


# ---------------------------------------------------------------- Mimi

def load_mimi(device="cuda", source="kyutai/mimi"):
    from transformers import AutoFeatureExtractor, MimiModel

    model = MimiModel.from_pretrained(source).to(device).eval()
    fe = AutoFeatureExtractor.from_pretrained(source)
    return model, fe


def mimi_features(model, fe, wav24k, n_q=8):
    """Encode and return frame-level continuous features, each [T, 512] float32.

      codes     [8, T]   RVQ indices
      semantic  CB1 decoded through Mimi's own codebook + output projection
      acoustic  CB2..CB8 decoded the same way (the part BayLing-MP would add)
      all       CB1..CB8 (= what Moshi reads)
    """
    import torch

    inputs = fe(raw_audio=np.asarray(wav24k, dtype=np.float32), sampling_rate=MIMI_SR,
                return_tensors="pt")
    x = inputs["input_values"].to(model.device)
    with torch.no_grad():
        codes = model.encode(x, num_quantizers=n_q).audio_codes  # [1, 8, T]
        q = model.quantizer
        n_sem = q.num_semantic_quantizers
        sem = q.semantic_residual_vector_quantizer.decode(codes[:, :n_sem])     # [1, 512, T]
        aco = q.acoustic_residual_vector_quantizer.decode(codes[:, n_sem:])     # [1, 512, T]
    to_np = lambda t: t[0].transpose(0, 1).float().cpu().numpy()  # noqa: E731
    semantic, acoustic = to_np(sem), to_np(aco)
    return {
        "codes": codes[0].cpu().numpy(),
        "semantic": semantic,
        "acoustic": acoustic,
        "all": semantic + acoustic,
    }
