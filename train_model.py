"""Train and evaluate the CPU-only EOT model."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eot_features import extract_features, load_wav


def read_datasets(data_dirs: list[Path]):
    X, y, groups, keys, languages = [], [], [], [], []
    for data_dir in data_dirs:
        rows = list(csv.DictReader((data_dir / "labels.csv").open()))
        cache = {}
        for row in rows:
            audio_path = data_dir / row["audio_file"]
            if audio_path not in cache:
                cache[audio_path] = load_wav(audio_path)
            x, sr = cache[audio_path]
            X.append(extract_features(x, sr, float(row["pause_start"]),
                                      int(row["pause_index"])))
            y.append(row["label"] == "eot")
            group = f"{data_dir.resolve()}::{row['turn_id']}"
            groups.append(group)
            keys.append((str(data_dir), row["turn_id"], row["pause_index"]))
            languages.append(data_dir.name)
    return np.vstack(X), np.asarray(y, dtype=np.int8), np.asarray(groups), keys, languages


def make_model(seed: int = 42):
    linear = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(C=0.25, class_weight="balanced", max_iter=3000,
                           random_state=seed),
    )
    trees = ExtraTreesClassifier(
        n_estimators=500, min_samples_leaf=4, max_features=0.7,
        class_weight="balanced", random_state=seed, n_jobs=1,
    )
    return VotingClassifier(
        estimators=[("linear", linear), ("trees", trees)],
        voting="soft", weights=[1.0, 4.0], n_jobs=1,
    )


def write_predictions(path: Path, keys, probabilities, data_dir: Path):
    wanted = str(data_dir)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["turn_id", "pause_index", "p_eot"])
        for key, prob in zip(keys, probabilities):
            if key[0] == wanted:
                writer.writerow([key[1], key[2], f"{float(prob):.6f}"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--model_out", type=Path, default=Path("eot_model.joblib"))
    parser.add_argument("--oof_prefix", default="oof")
    args = parser.parse_args()

    X, y, groups, keys, languages = read_datasets(args.data_dirs)
    print(f"examples={len(y)} features={X.shape[1]} turns={len(set(groups))} "
          f"eot={int(y.sum())} hold={int((1-y).sum())}")
    cv = GroupKFold(n_splits=5)
    oof = cross_val_predict(make_model(), X, y, groups=groups, cv=cv,
                            method="predict_proba", n_jobs=1)[:, 1]
    for data_dir in args.data_dirs:
        out = Path(f"{args.oof_prefix}_{data_dir.name}.csv")
        write_predictions(out, keys, oof, data_dir)
        print(f"wrote grouped OOF predictions -> {out}")

    model = make_model()
    model.fit(X, y)
    artifact = {
        "model": model,
        "feature_version": 1,
        "feature_count": int(X.shape[1]),
        "training_languages": sorted(set(languages)),
    }
    joblib.dump(artifact, args.model_out, compress=3)
    print(f"wrote final model -> {args.model_out}")


if __name__ == "__main__":
    main()
