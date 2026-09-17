"""
Offline checks (numpy only):  python -m unittest discover -s tests -v
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spkprobe import analyze, fake_features  # noqa: E402
from spkprobe.check_tts import centroid_loo_accuracy, wer  # noqa: E402
from spkprobe.common import complete_grid, read_sentences, trim_silence  # noqa: E402
from spkprobe.extract_bayling import (build_interleaved_ids, interleaved_user_positions,  # noqa: E402
                                      prepare_waveform, speech_frame_mask)


class LayoutTest(unittest.TestCase):
    def test_user_positions_match_interleaving(self):
        n_blocks = 3
        user = [f"u{i}" for i in range(30)]
        seq = build_interleaved_ids(user, ["t"] * 15, ["a"] * 30)
        self.assertEqual(len(seq), 75)
        pos = interleaved_user_positions(n_blocks)
        self.assertEqual([seq[p] for p in pos], user)

    def test_speech_mask(self):
        # lead 1.6 s = 20 frames; speech 0.4 s = 5 frames
        m = speech_frame_mask(40, lead_s=1.6, speech_s=0.4)
        self.assertEqual(m.sum(), 5)
        self.assertTrue(m[20:25].all())

    def test_prepare_waveform_block_aligned(self):
        sr = 16000
        x = np.zeros(sr * 3, np.float32)
        x[sr:2 * sr] = 0.5 * np.sin(np.arange(sr) * 0.05)
        wave, lead_s, speech_s, n_blocks = prepare_waveform(x, sr, lead_blocks=2)
        self.assertAlmostEqual(lead_s, 1.6)
        self.assertEqual(len(wave), n_blocks * int(0.8 * sr))
        self.assertTrue(0.95 < speech_s < 1.1)
        # speech must start exactly after the lead silence
        self.assertEqual(float(np.abs(wave[:int(1.6 * sr)]).max()), 0.0)


class CommonTest(unittest.TestCase):
    def test_complete_grid_drops_missing_sentence(self):
        spk = ["a", "a", "a", "b", "b", "c", "c", "c"]
        sent = ["1", "2", "3", "1", "2", "1", "2", "3"]
        mask, spks, sents = complete_grid(spk, sent)
        self.assertEqual(spks, ["a", "b", "c"])
        self.assertEqual(sents, ["1", "2"])
        self.assertEqual(mask.sum(), 6)

    def test_harvard_file(self):
        rows = read_sentences(Path(__file__).resolve().parents[1] / "data" / "harvard_sentences.txt")
        self.assertEqual(len(rows), 40)
        self.assertEqual(len({t for _, t in rows}), 40)

    def test_trim_silence(self):
        sr = 16000
        x = np.zeros(sr, np.float32)
        x[4000:8000] = 0.3
        y, s, e = trim_silence(x, sr)
        self.assertTrue(3800 <= s <= 4000 and 8000 <= e <= 8600)


class QcTest(unittest.TestCase):
    def test_wer(self):
        self.assertEqual(wer("the birch canoe", "the birch canoe"), 0.0)
        self.assertAlmostEqual(wer("the birch canoe slid", "the canoe slid"), 0.25)
        self.assertAlmostEqual(wer("a b", "a b c d"), 1.0)

    def test_centroid_accuracy_separable(self):
        rng = np.random.default_rng(0)
        centers = rng.normal(size=(4, 16)) * 5
        emb = np.concatenate([c + rng.normal(size=(10, 16)) for c in centers])
        labels = np.repeat(np.arange(4), 10)
        self.assertGreater(centroid_loo_accuracy(emb, labels), 0.95)


class AnalysisTest(unittest.TestCase):
    def test_group_folds_keep_groups_apart(self):
        groups = np.repeat(np.arange(10), 3)
        for tr, te in analyze.group_folds(groups, 5):
            self.assertFalse(set(groups[tr]) & set(groups[te]))

    def test_recovers_planted_peak(self):
        feats, ecapa, utts, spk, sent = fake_features.make(peak_layer=8, seed=1)
        accs = [analyze.probe_accuracy(feats[:, l], spk, sent, n_splits=5)[0] for l in range(feats.shape[1])]
        idx = [analyze.similarity_index(feats[:, l], spk, sent)[2] for l in range(feats.shape[1])]
        self.assertIn(int(np.argmax(accs)), (7, 8, 9))
        self.assertIn(int(np.argmax(idx)), (7, 8, 9))
        self.assertLess(accs[0], 0.5)          # layer 0 is content-dominated
        self.assertLess(idx[0], 0.0)

    def test_speaker_probe_has_no_content_leak(self):
        # features carry only sentence identity: speaker probe must stay near chance
        rng = np.random.default_rng(2)
        sent_vec = rng.normal(size=(20, 32)) * 3
        spk = np.repeat([f"s{i}" for i in range(8)], 20)
        sent = np.tile([f"t{j}" for j in range(20)], 8)
        X = sent_vec[np.tile(np.arange(20), 8)] + rng.normal(size=(160, 32))
        acc, _ = analyze.probe_accuracy(X, spk, sent)
        self.assertLess(acc, 0.3)

    def test_end_to_end_cli(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            sys.argv = ["fake_features", "--out", str(d)]
            fake_features.main()
            sys.argv = ["analyze", "--bayling", str(d / "bayling_fake.npz"),
                        "--baselines", str(d / "baselines.npz"), "--out", str(d / "res"), "--folds", "4"]
            analyze.main()
            text = (d / "res" / "metrics.csv").read_text()
            self.assertIn("bayling_fake", text)
            self.assertIn("ecapa_fake", text)
            self.assertTrue((d / "res" / "summary.md").exists())


if __name__ == "__main__":
    unittest.main()
