# CPU-only End-of-Turn Detector

The supplied `eot_data/` call recordings are intentionally excluded from version control. Place the assignment-provided data folder beside `predict.py` before running the reproduction commands below; the expected structure is `eot_data/english/{labels.csv,audio/}` and `eot_data/hindi/{labels.csv,audio/}`.

## Run the required predictor

Activate the prepared environment, then run from this directory:

```bash
python predict.py --data_dir eot_data/english --out predictions.csv
```

`predict.py` loads `eot_model.joblib`; it never fits on the target folder. The output schema is exactly `turn_id,pause_index,p_eot`.

## Reproduce training and grouped validation

```bash
python train_model.py \
  --data_dirs eot_data/english eot_data/hindi \
  --model_out eot_model.joblib \
  --oof_prefix oof_final

python starter/score.py --data_dir eot_data/english --pred oof_final_english.csv
python starter/score.py --data_dir eot_data/hindi --pred oof_final_hindi.csv
```

## Validate the packaged submission

```bash
python test_causality.py

python predict.py \
  --data_dir eot_data/english \
  --out predictions_english.csv
python predict.py \
  --data_dir eot_data/hindi \
  --out predictions_hindi.csv

python starter/score.py \
  --data_dir eot_data/english \
  --pred predictions_english.csv
python starter/score.py \
  --data_dir eot_data/hindi \
  --pred predictions_hindi.csv
```

The causality test checks all 496 supplied annotations: for each pause it changes every sample at and after `pause_start` and verifies that the extracted feature vector is unchanged.
