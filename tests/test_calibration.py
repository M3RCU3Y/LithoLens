import numpy as np

from litholens.mvp_baseline import expected_calibration_error


def test_expected_calibration_error_is_zero_for_perfect_confidence():
    confidence = np.array([1.0, 1.0, 1.0])
    correct = np.array([True, True, True])

    assert expected_calibration_error(confidence, correct, n_bins=5) == 0.0


def test_expected_calibration_error_increases_when_confidence_misleads():
    confidence = np.array([0.9, 0.9, 0.9, 0.9])
    correct = np.array([True, False, False, False])

    assert expected_calibration_error(confidence, correct, n_bins=5) > 0.5
