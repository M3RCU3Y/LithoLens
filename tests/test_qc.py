import pandas as pd

from litholens.qc import detect_spikes


def test_spike_detection_catches_injected_spike():
    df = pd.DataFrame({"GR": [80.0] * 20 + [500.0] + [81.0] * 20})
    flags = detect_spikes(df, ["GR"], z_threshold=6.0)
    assert flags["GR_spike_flag"].iloc[20]
    assert flags["GR_spike_flag"].sum() == 1
