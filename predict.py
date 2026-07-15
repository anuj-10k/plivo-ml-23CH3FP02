"""Generate causal end-of-turn probabilities for an unseen data folder.

Usage: python predict.py --data_dir <folder> --out predictions.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import joblib
import numpy as np

from eot_features import extract_features, load_wav


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("predictions.csv"))
    parser.add_argument("--model", type=Path,
                        default=Path(__file__).with_name("eot_model.joblib"))
    args = parser.parse_args()

    rows = list(csv.DictReader((args.data_dir / "labels.csv").open()))
    artifact = joblib.load(args.model)
    cache = {}
    features = []
    for row in rows:
        wav_path = args.data_dir / row["audio_file"]
        if wav_path not in cache:
            cache[wav_path] = load_wav(wav_path)
        x, sr = cache[wav_path]
        features.append(extract_features(x, sr, float(row["pause_start"]),
                                         int(row["pause_index"])))
    X = np.vstack(features)
    if X.shape[1] != artifact["feature_count"]:
        raise RuntimeError("feature/model version mismatch")
    probabilities = artifact["model"].predict_proba(X)[:, 1]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["turn_id", "pause_index", "p_eot"])
        for row, probability in zip(rows, probabilities):
            writer.writerow([row["turn_id"], row["pause_index"],
                             f"{float(probability):.6f}"])
    print(f"wrote {len(rows)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
