import numpy as np
import pandas as pd

from litholens.mvp_baseline import force_penalty_score


def test_force_penalty_score_uses_force_label_order():
    matrix = np.arange(144).reshape(12, 12)
    truth = pd.Series([30000, 65000, 93000])
    pred = pd.Series([65030, 65000, 30000])

    score = force_penalty_score(truth, pred, matrix)

    expected = (matrix[0, 1] + matrix[2, 2] + matrix[11, 0]) / 3
    assert score == expected
