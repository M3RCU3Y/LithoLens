"""Run the LithoLens baseline workflow."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from litholens.pipeline import run_training_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LithoLens baseline training.")
    parser.add_argument("--config", default="configs/baseline.yaml", help="Path to YAML config.")
    args = parser.parse_args()
    result = run_training_pipeline(Path(args.config))
    print("Baseline complete")
    print(f"Model: {result['model_path']}")
    print(f"Predictions: {result['prediction_path']}")
    print(f"Metrics: {result['metrics']}")


if __name__ == "__main__":
    main()
