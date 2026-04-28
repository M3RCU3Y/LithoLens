import numpy as np

from litholens.uncertainty import shannon_entropy


def test_entropy_uniform_is_higher_than_certain():
    uniform = shannon_entropy(np.array([[0.25, 0.25, 0.25, 0.25]]))[0]
    certain = shannon_entropy(np.array([[1.0, 0.0, 0.0, 0.0]]))[0]
    assert uniform > 1.3
    assert certain < 0.01
