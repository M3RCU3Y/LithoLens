# LithoLens Baseline Model Card

## Model

Initial baseline: scikit-learn `RandomForestClassifier` trained on standardized well-log curves, missingness indicators, rolling features, gradients, and selected cross-log ratios.

## Intended Use

Assist lithology interpretation by producing a predicted lithology per depth sample with confidence, uncertainty flags, QC warnings, and plain-language explanations. The model is a decision-support tool and does not replace geoscientist review.

## Evaluation

Final metrics must use well-level holdout or GroupKFold splits. Random row-level train/test splits are not acceptable for final reporting because adjacent depth samples and the same well can leak geological context.

## Limitations

- Performance depends on curve coverage and label quality.
- Rare lithology classes may be underrepresented.
- Confidence may be miscalibrated until calibration is added.
- Imputation flags must be reviewed because filled values are not measurements.

## Review Guidance

Intervals with low confidence, low probability margin, high entropy, missing/imputed curves, spikes, or range violations should be treated as review zones.
