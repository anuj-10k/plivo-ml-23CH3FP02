"""Regression checks for the hard causality boundary and output schema."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from eot_features import extract_features, load_wav


ROOT = Path(__file__).resolve().parent


def test_future_audio_is_ignored() -> None:
    rng = np.random.default_rng(7)
    checked = 0
    for language in ("english", "hindi"):
        data_dir = ROOT / "eot_data" / language
        rows = list(csv.DictReader((data_dir / "labels.csv").open()))
        cache = {}
        for row in rows:
            wav_path = data_dir / row["audio_file"]
            if wav_path not in cache:
                cache[wav_path] = load_wav(wav_path)
            x, sr = cache[wav_path]
            pause_start = float(row["pause_start"])
            pause_index = int(row["pause_index"])
            original = extract_features(x, sr, pause_start, pause_index)
            changed = x.copy()
            boundary = min(len(x), int(np.floor(pause_start * sr)))
            changed[boundary:] = rng.uniform(-1.0, 1.0,
                                             len(changed) - boundary)
            modified = extract_features(changed, sr, pause_start, pause_index)
            np.testing.assert_array_equal(original, modified)
            checked += 1
    assert checked == 496, f"expected 496 annotated pauses, checked {checked}"


def main() -> None:
    test_future_audio_is_ignored()
    print("OK: future-audio invariance holds for all 496 annotated pauses")


if __name__ == "__main__":
    main()
