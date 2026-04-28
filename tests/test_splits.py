import pandas as pd

from litholens.split import get_group_kfold_splits


def test_group_kfold_does_not_mix_wells_between_train_and_validation():
    df = pd.DataFrame({"WELL": ["A"] * 3 + ["B"] * 3 + ["C"] * 3, "GR": range(9)})
    splits = get_group_kfold_splits(df, "WELL", n_splits=3)
    for train_idx, val_idx in splits:
        train_wells = set(df.iloc[train_idx]["WELL"])
        val_wells = set(df.iloc[val_idx]["WELL"])
        assert train_wells.isdisjoint(val_wells)
