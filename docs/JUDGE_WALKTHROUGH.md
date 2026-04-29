# LithoLens Judge Walkthrough

## One-Sentence Pitch

LithoLens predicts lithology from well logs and highlights where the model should be trusted, questioned, or reviewed by a geoscientist.

## Demo Setup

Run:

```powershell
python scripts/run_baseline.py --config configs/baseline.yaml --input data/raw/force_train.csv --output-dir reports/mvp_baseline --trees 50 --compare-models
streamlit run app/streamlit_app.py
```

## Talk Track

1. **Problem:** lithology interpretation is repetitive, data quality varies by well, and a black-box label is not enough for operational use.
2. **Data:** FORCE 2020 well logs provide a public benchmark with lithology labels and an official penalty matrix.
3. **Validation:** LithoLens uses GroupKFold by well, so validation wells are not seen during training.
4. **Prediction:** the dashboard shows raw logs beside predicted lithology for the held-out well.
5. **Trust layer:** each interval includes confidence, margin, entropy, QC warnings, and a review-zone flag.
6. **Human-in-the-loop:** review zones are where the interpreter should slow down instead of accepting the model blindly.
7. **Evidence:** show fold metrics, FORCE penalty score, calibration bins, model comparison, and explanations.

## What To Click

- **Tracks:** show raw curves, predicted lithology, confidence, uncertainty, and QC track.
- **Review Zones:** sort through intervals with missing curves, range warnings, spike warnings, or high uncertainty.
- **Explanations:** pick a depth and read the plain-language prediction note.
- **Metrics:** show all GroupKFold fold results.
- **Calibration:** show whether confidence is aligned with empirical accuracy.
- **Model Comparison:** show why RandomForest remains the reliable MVP baseline.

## Strong Closing

LithoLens is not claiming to replace the interpreter. It turns a lithology classifier into a reviewable interpretation assistant with uncertainty, data-quality context, and reproducible validation.
