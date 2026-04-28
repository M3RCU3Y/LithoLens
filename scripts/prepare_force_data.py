"""Prepare local FORCE 2020 training data from cloned references."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def prepare_force_data(
    zip_path: Path = Path("references/force-2020-official/lithology_competition/data/train.zip"),
    output_path: Path = Path("data/raw/force_train.csv"),
) -> Path:
    """Extract the official FORCE train CSV into the local ignored data folder."""
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Missing {zip_path}. Run scripts/clone_references.sh or clone the official FORCE repo first."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("train.csv") as source, output_path.open("wb") as destination:
            destination.write(source.read())
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract FORCE 2020 train.csv locally.")
    parser.add_argument("--zip-path", type=Path, default=Path("references/force-2020-official/lithology_competition/data/train.zip"))
    parser.add_argument("--output", type=Path, default=Path("data/raw/force_train.csv"))
    args = parser.parse_args()
    path = prepare_force_data(args.zip_path, args.output)
    print(f"Prepared FORCE training CSV: {path}")


if __name__ == "__main__":
    main()
