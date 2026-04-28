"""Shared constants and conservative defaults for LithoLens."""

COMMON_LOG_COLUMNS = ["GR", "RHOB", "NPHI", "DTC", "RDEP", "RMED", "CALI", "PEF", "SP"]

DEPTH_COLUMN_CANDIDATES = ["DEPTH_MD", "DEPTH", "DEPT", "MD"]
TARGET_COLUMN_CANDIDATES = [
    "FORCE_2020_LITHOFACIES_LITHOLOGY",
    "LITHOLOGY",
    "FACIES",
    "TARGET",
]
WELL_COLUMN_CANDIDATES = ["WELL", "WELL_NAME", "WELL_ID", "BOREHOLE"]

MNEMONIC_ALIASES = {
    "DEPTH": "DEPTH_MD",
    "DEPT": "DEPTH_MD",
    "MD": "DEPTH_MD",
    "WELL_NAME": "WELL",
    "WELL_ID": "WELL",
    "BOREHOLE": "WELL",
    "LITHOLOGY": "FORCE_2020_LITHOFACIES_LITHOLOGY",
    "FACIES": "FORCE_2020_LITHOFACIES_LITHOLOGY",
    "TARGET": "FORCE_2020_LITHOFACIES_LITHOLOGY",
    "GAMMA": "GR",
    "GAMMA_RAY": "GR",
    "DENSITY": "RHOB",
    "NEUTRON": "NPHI",
    "SONIC": "DTC",
    "DEEP_RESISTIVITY": "RDEP",
    "MEDIUM_RESISTIVITY": "RMED",
}

FORCE_LITHOLOGY_LABELS = {
    30000: "Sandstone",
    65030: "Sandstone/Shale",
    65000: "Shale",
    80000: "Marl",
    74000: "Dolomite",
    70000: "Limestone",
    70032: "Chalk",
    88000: "Halite",
    86000: "Anhydrite",
    99000: "Tuff",
    90000: "Coal",
    93000: "Basement",
}

DEFAULT_PHYSICAL_RANGES = {
    "GR": (0.0, 350.0),
    "RHOB": (1.0, 3.5),
    "NPHI": (-0.15, 1.0),
    "DTC": (20.0, 250.0),
    "RDEP": (0.01, 100000.0),
    "RMED": (0.01, 100000.0),
    "CALI": (2.0, 40.0),
    "PEF": (0.0, 20.0),
    "SP": (-500.0, 500.0),
}
