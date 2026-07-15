# End-of-Turn Model Run Log

All development estimates below are grouped out-of-fold (OOF) predictions: all pauses from a turn stay in the same fold.  The only exception is Run 6, explicitly marked as a training-set diagnostic for the required supplied-data CSVs.

## Run 1 — silence-only baseline

| Data | Mean delay | Interrupted turns | AUC | Operating point |
|---|---:|---:|---:|---|
| English | 1600 ms | 0.0% | 0.506 | threshold 1.00, delay 1600 ms |
| Hindi | 850 ms | 5.0% | 0.516 | threshold 0.05, delay 850 ms |

Every pause received `p_eot=1`; this establishes the supplied VAD-style reference. Hindi has no hold pause longer than 1.5 s, so its baseline can safely wait 850 ms and lands exactly on the 5% cutoff budget.

## Run 2 — three-model causal feature ensemble (`oof_v1_*`)

| Data | Mean delay | Interrupted turns | AUC | Operating point |
|---|---:|---:|---:|---|
| English | 1173 ms | 5.0% | 0.684 | threshold 0.45, delay 900 ms |
| Hindi | 850 ms | 5.0% | 0.792 | threshold 0.05, delay 850 ms |

Added 136 causal features over 0.3, 0.7, 1.5, and 3.0 s windows and blended logistic regression, Extra Trees, and histogram gradient boosting. This substantially improves ranking in both languages and beats the English baseline.

## Run 3 — logistic regression ablation (`oof_logistic_*`)

| Data | Mean delay | Interrupted turns | AUC | Operating point |
|---|---:|---:|---:|---|
| English | 1286 ms | 5.0% | 0.654 | threshold 0.70, delay 650 ms |
| Hindi | 850 ms | 5.0% | 0.726 | threshold 0.05, delay 850 ms |

Tested whether a regularized linear decision surface generalizes better on this small dataset. It is stable but misses useful nonlinear interactions among pitch, energy, and turn position.

## Run 4 — Extra Trees ablation (`oof_trees_*`)

| Data | Mean delay | Interrupted turns | AUC | Operating point |
|---|---:|---:|---:|---|
| English | 1110 ms | 5.0% | 0.687 | threshold 0.50, delay 600 ms |
| Hindi | 850 ms | 5.0% | 0.789 | threshold 0.05, delay 850 ms |

Extra Trees gives the best single-model operational result, showing that nonlinear feature combinations are important. A minimum leaf size of four limits memorization.

## Run 5 — histogram gradient boosting ablation (`oof_hist_*`)

| Data | Mean delay | Interrupted turns | AUC | Operating point |
|---|---:|---:|---:|---|
| English | 1250 ms | 5.0% | 0.656 | threshold 0.25, delay 1100 ms |
| Hindi | 857 ms | 5.0% | 0.757 | threshold 0.05, delay 850 ms |

Tested a compact boosted model as a lower-variance nonlinear alternative. It underperforms Extra Trees and is excluded from the final artifact.

## Run 6 — final 4:1 Extra Trees/logistic blend

| Evaluation | Data | Mean delay | Interrupted turns | AUC | Operating point |
|---|---|---:|---:|---:|---|
| 5-fold grouped OOF | English | **1078 ms** | 5.0% | 0.691 | threshold 0.50, delay 650 ms |
| 5-fold grouped OOF | Hindi | **850 ms** | 5.0% | 0.790 | threshold 0.05, delay 850 ms |
| Fitted-data diagnostic | English | 100 ms | 5.0% | 1.000 | threshold 0.40, delay 100 ms |
| Fitted-data diagnostic | Hindi | 100 ms | 5.0% | 1.000 | threshold 0.35, delay 100 ms |

The final weighting keeps the strong tree boundary and uses a smaller linear component for smoothing. Model choice is based on grouped OOF results; the fitted-data numbers are reported only because `predictions_english.csv` and `predictions_hindi.csv` are required deliverables and must not be interpreted as unseen performance.

## Run 7 — final submission verification rerun

| Evaluation | Data | Mean delay | Interrupted turns | AUC | Operating point |
|---|---|---:|---:|---:|---|
| 5-fold grouped OOF | English | **1078 ms** | 5.0% | 0.691 | threshold 0.50, delay 650 ms |
| 5-fold grouped OOF | Hindi | **850 ms** | 5.0% | 0.790 | threshold 0.05, delay 850 ms |
| Required fitted-data CSV | English | 100 ms | 5.0% | 1.000 | threshold 0.40, delay 100 ms |
| Required fitted-data CSV | Hindi | 100 ms | 5.0% | 1.000 | threshold 0.35, delay 100 ms |

No model or feature change was made in this run. The official scorer was rerun after the submission-readiness audit to verify that documentation and test improvements did not alter either the honest OOF results or the required prediction files; the 100 ms results remain fitted-data diagnostics, not unseen-data estimates.
