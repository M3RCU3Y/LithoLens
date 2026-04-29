"""Run the LithoLens baseline workflow."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from litholens.mvp_baseline import run_mvp_baseline
from litholens.pipeline import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LithoLens baseline training.")
    parser.add_argument("--config", default="configs/baseline.yaml", help="Path to YAML config.")
    parser.add_argument("--input", help="Optional FORCE-style CSV path. Overrides config data.raw_path.")
    parser.add_argument("--output-dir", default="reports/mvp_baseline", help="Output directory.")
    parser.add_argument("--trees", type=int, default=100, help="RandomForest tree count.")
    parser.add_argument("--compare-models", action="store_true", help="Export first-fold model comparison.")
    args = parser.parse_args()
    config = load_config(Path(args.config))
    input_path = Path(args.input or config["data"]["raw_path"])
    result = run_mvp_baseline(
        input_path=input_path,
        output_dir=Path(args.output_dir),
        n_estimators=args.trees,
        compare_models=args.compare_models,
    )
    print("MVP baseline complete")
    print(f"Held-out well: {result.heldout_well}")
    print(f"Weighted F1: {result.weighted_f1:.4f}")
    print(f"Model: {result.model_path}")
    print(f"Confusion matrix: {result.confusion_matrix_path}")
    print(f"Predictions: {result.predictions_path}")


if __name__ == "__main__":
    main()
