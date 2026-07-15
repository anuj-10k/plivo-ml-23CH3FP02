# Final Results — End-of-Turn Detection

## Final outcome

The final system is a CPU-only causal end-of-turn classifier trained on the supplied English and Hindi data. It produces one `p_eot` probability for every annotated pause and never uses audio at or after `pause_start`.

The strongest honest validation result was:

- **English: 1078 ms mean response delay at 5.0% interrupted turns**, compared with the 1600 ms silence-only baseline.
- **Hindi: 850 ms mean response delay at 5.0% interrupted turns**, tying the unusually strong Hindi silence baseline while improving AUC from 0.516 to 0.790.
- English response delay improved by **522 ms**, or approximately **32.6%**.

## Score summary

### Silence-only baseline

| Language | Mean response delay | Interrupted turns | AUC | Operating point |
|---|---:|---:|---:|---|
| English | 1600 ms | 0.0% | 0.506 | threshold 1.00, delay 1600 ms |
| Hindi | 850 ms | 5.0% | 0.516 | threshold 0.05, delay 850 ms |

### Final model — grouped out-of-fold validation

These are the meaningful generalization estimates. Five-fold validation kept every pause belonging to the same turn entirely inside either training or validation.

| Language | Mean response delay | Interrupted turns | AUC | Operating point |
|---|---:|---:|---:|---|
| English | **1078 ms** | 5.0% | 0.691 | threshold 0.50, delay 650 ms |
| Hindi | **850 ms** | 5.0% | 0.790 | threshold 0.05, delay 850 ms |

### Required supplied-data prediction files

The final model artifact was fitted on all supplied English and Hindi examples before generating the required prediction CSVs.

| Prediction file | Mean response delay | Interrupted turns | AUC | Operating point |
|---|---:|---:|---:|---|
| `predictions_english.csv` | 100 ms | 5.0% | 1.000 | threshold 0.40, delay 100 ms |
| `predictions_hindi.csv` | 100 ms | 5.0% | 1.000 | threshold 0.35, delay 100 ms |

The 100 ms results above are training-set diagnostics because these rows were included when fitting the final artifact. They confirm that the required files are valid, but the grouped out-of-fold scores are the correct estimates of performance on unseen turns.

## What was implemented

1. Read the assignment, setup instructions, starter code, scoring logic, and completion plan.
2. Ran and recorded the silence-only baseline for both languages.
3. Analyzed 200 turns and 496 annotated pauses: 100 turns, 148 hold pauses, and 100 EOT pauses per language.
4. Implemented 136 causal features over 0.3, 0.7, 1.5, and 3.0 second context windows.
5. Extracted energy, energy-decay, pitch, pitch-slope, voicing, final voiced-region, zero-crossing, rhythm-proxy, elapsed-context, and pause-index features.
6. Compared logistic regression, Extra Trees, histogram gradient boosting, and blended models using grouped validation.
7. Selected a soft-voting model containing four parts Extra Trees and one part balanced logistic regression.
8. Trained the final model on the combined English and Hindi datasets and saved it as `eot_model.joblib`.
9. Created a `predict.py` that loads the saved model and never trains on the target folder.
10. Generated the required English and Hindi prediction CSV files and checked them with the official scorer.
11. Added an automated causality test that changes all future audio and confirms that the feature vector remains identical.
12. Created the HTML summary, full run log, short notes, reproduction instructions, and dependency list.

## Causality guarantee

All acoustic features are computed only after slicing the waveform at:

```python
past = x[:floor(pause_start * sample_rate)]
```

The feature extractor does not accept `pause_end` or pause duration. For all 496 supplied annotations, `test_causality.py` replaces every sample at and after `pause_start` with random noise and verifies that all extracted features remain bit-for-bit unchanged.

## Final files

| File | Purpose |
|---|---|
| `predict.py` | Required inference entry point |
| `eot_model.joblib` | Saved fitted model used by `predict.py` |
| `eot_features.py` | Causal audio feature extraction |
| `train_model.py` | Reproducible grouped validation and training |
| `predictions_english.csv` | Required English predictions |
| `predictions_hindi.csv` | Required Hindi predictions |
| `oof_final_english.csv` | Honest grouped-validation English predictions |
| `oof_final_hindi.csv` | Honest grouped-validation Hindi predictions |
| `SUMMARY.html` | Detailed visual solution report |
| `RUNLOG.md` | Results from every scoring experiment |
| `NOTES.md` | Required short model notes |
| `test_causality.py` | Automated future-audio leakage check |
| `README.md` | Usage and reproduction instructions |
| `requirements.txt` | Runtime dependencies |

## Commands to check the score

From the `eot_handout` directory, with the prepared Python environment activated:

```bash
# Honest grouped-validation scores
python starter/score.py --data_dir eot_data/english --pred oof_final_english.csv
python starter/score.py --data_dir eot_data/hindi --pred oof_final_hindi.csv

# Scores for the required final prediction files
python starter/score.py --data_dir eot_data/english --pred predictions_english.csv
python starter/score.py --data_dir eot_data/hindi --pred predictions_hindi.csv
```

## Commands to verify inference

```bash
python test_causality.py
python predict.py --data_dir eot_data/english --out check_predictions.csv
python starter/score.py --data_dir eot_data/english --pred check_predictions.csv
```

Expected causality-test output:

```text
OK: future-audio invariance holds for all 496 annotated pauses
```

## Conclusion

The final system clearly beats silence-only endpointing on English by using already-spoken prosodic evidence instead of treating every pause identically. Hindi classification ranking also improves substantially, although the scorer's discrete operating-point sweep leaves its delay tied with the favorable Hindi silence baseline. The complete solution runs locally on CPU, uses no pretrained models or external data, respects the assignment's hard causality rule, and includes all required deliverables.
