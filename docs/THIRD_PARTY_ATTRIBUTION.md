# Third-Party Attribution

This project studies public repositories and datasets for hackathon context. External repositories must be cloned into `references/` only and must not be merged directly into `src/`.

## Planned References

| Source | URL | Intended Use | Integration Rule |
| --- | --- | --- | --- |
| FORCE 2020 Machine Learning Competition | https://github.com/bolgebrygg/Force-2020-Machine-Learning-competition | Dataset format, starter notebook, lithology labels, penalty matrix, competition framing | Study only; cite any copied label mappings or scoring details |
| Uncertainty-aware facies prediction | https://github.com/artem-potlog/well-log-facies-prediction | Uncertainty, calibration, validation, feature engineering inspiration | Study only; no direct code copy |
| FORCE 2020 EDA | https://github.com/mycarta/Force-2020-Machine-Learning-competition_predict-lithology-EDA | Well-log visualization and EDA patterns | Study only; no direct code copy |
| FORCE solution reference | https://github.com/olawaleibrahim/2020_FORCE_Lithology_Prediction | Modeling inspiration and benchmark comparison | Study only; no direct code copy |

## Dataset Attribution

FORCE 2020 lithofacies data should be attributed to the public FORCE 2020 well-log lithofacies machine learning competition and associated dataset publication. Add the exact dataset citation and license after the team confirms the downloaded data source.

## Confirmed FORCE Reference Details

- Official starter notebook label map:
  - `30000`: Sandstone
  - `65030`: Sandstone/Shale
  - `65000`: Shale
  - `80000`: Marl
  - `74000`: Dolomite
  - `70000`: Limestone
  - `70032`: Chalk
  - `88000`: Halite
  - `86000`: Anhydrite
  - `99000`: Tuff
  - `90000`: Coal
  - `93000`: Basement
- The official penalty matrix is present at `references/force-2020-official/lithology_competition/data/penalty_matrix.npy`.
- The penalty matrix shape is `12 x 12`, matching the 12 FORCE lithology classes.

## License Notes

Before copying any code, label mappings, penalty matrices, or notebook snippets from a reference repo, check its license and add a specific attribution entry here. Prefer reimplementation from first principles.
