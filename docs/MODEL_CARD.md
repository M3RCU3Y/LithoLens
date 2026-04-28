# LithoLens Baseline Model Card

## Model

Current MVP baseline: scikit-learn `RandomForestClassifier` trained on standardized FORCE-style well-log curves and missingness indicators.

## Intended Use

Assist lithology interpretation by producing a predicted lithology per depth sample with confidence, uncertainty flags, QC warnings, and plain-language explanations. The model is a decision-support tool and does not replace geoscientist review.

## Evaluation

Metrics use well-level `GroupKFold` splits. Random row-level train/test splits are not acceptable for final reporting because adjacent depth samples and the same well can leak geological context.

The MVP exports:

- per-fold weighted F1
- per-fold FORCE penalty score when `penalty_matrix.npy` is available
- aggregate confusion matrix
- out-of-fold predictions
- one held-out well prediction file for the demo

## Limitations

- Performance depends on curve coverage and label quality.
- Rare lithology classes may be underrepresented.
- Confidence may be miscalibrated until calibration is added.
- Imputation flags must be reviewed because filled values are not measurements.
- Current explanations use RandomForest feature importances, not SHAP.

## Review Guidance

Intervals with low confidence, low probability margin, high entropy, missing/imputed curves, spikes, or range violations should be treated as review zones.
