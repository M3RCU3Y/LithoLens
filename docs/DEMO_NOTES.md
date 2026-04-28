# LithoLens Demo Notes

## Demo Flow

1. Run the MVP baseline:

   ```powershell
   python scripts/run_baseline.py --config configs/baseline.yaml --input data/raw/force_train.csv --output-dir reports/mvp_baseline --trees 50
   ```

2. Launch Streamlit:

   ```powershell
   streamlit run app/streamlit_app.py
   ```

3. Start on the held-out well view. Explain that the model has never trained on this well because validation is grouped by well.

4. Show the Tracks tab:
   - raw curves versus depth
   - predicted lithology track
   - confidence track
   - uncertainty and QC tracks

5. Show Review Zones:
   - low confidence
   - low probability margin
   - high entropy
   - missing/imputed log curves

6. Show Explanations:
   - prediction
   - confidence
   - dominant global feature drivers
   - QC warning if present

7. Show Fold Metrics:
   - weighted F1 by fold
   - FORCE penalty score by fold

## Current Limitation Script

LithoLens is a decision-support workflow, not a final geological interpretation. The current baseline is intentionally simple: RandomForest, common FORCE curves, missingness indicators, and no calibration yet. The important demo point is not just the raw score; it is that every prediction is paired with uncertainty, QC context, and human review guidance.

## Next Improvements

- Calibrate probabilities.
- Add richer QC: spikes, range violations, and missing intervals in the MVP artifact.
- Add SHAP or per-row local explanation once the baseline is stable.
- Compare RandomForest with HistGradientBoosting and LightGBM if available.
- Add LAS ingestion after FORCE CSV evaluation remains stable.
