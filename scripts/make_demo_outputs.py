"""Create demo outputs, using synthetic data if no raw CSV exists."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from litholens.pipeline import run_training_pipeline


def _synthetic_force_like(path: Path) -> None:
    rng = np.random.default_rng(42)
    rows = []
    labels = [30000, 65000, 70000]
    for well in ["DEMO_A", "DEMO_B", "DEMO_C", "DEMO_D", "DEMO_E"]:
        for i, depth in enumerate(np.arange(1000, 1100, 0.5)):
            lith = labels[(i // 35 + len(well)) % len(labels)]
            gr_center = {30000: 45, 65000: 115, 70000: 25}[lith]
            rows.append(
                {
                    "WELL": well,
                    "DEPTH_MD": depth,
                    "GR": rng.normal(gr_center, 8),
                    "RHOB": rng.normal(2.35 if lith != 65000 else 2.55, 0.06),
                    "NPHI": rng.normal(0.18 if lith != 70000 else 0.08, 0.03),
                    "DTC": rng.normal(85 if lith != 70000 else 65, 5),
                    "RDEP": abs(rng.normal(20 if lith == 70000 else 5, 2)),
                    "RMED": abs(rng.normal(15 if lith == 70000 else 4, 1.5)),
                    "CALI": rng.normal(8.5, 0.2),
                    "PEF": rng.normal(3.0 if lith != 70000 else 5.0, 0.3),
                    "SP": rng.normal(0, 15),
                    "FORCE_2020_LITHOFACIES_LITHOLOGY": lith,
                }
            )
    df = pd.DataFrame(rows)
    df.loc[df.sample(frac=0.02, random_state=7).index, "RDEP"] = np.nan
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Make LithoLens demo outputs.")
    parser.add_argument("--config", default="configs/baseline.yaml", help="Path to YAML config.")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw_path = Path(config["data"]["raw_path"])
    if not raw_path.exists():
        print(f"No training CSV found at {raw_path}; creating synthetic demo data.")
        _synthetic_force_like(raw_path)
    result = run_training_pipeline(config_path)
    print(f"Demo outputs ready: {result['prediction_path']}")


if __name__ == "__main__":
    main()
