from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class cfg:
    lag: int = 14
    temporal_split: bool = True
    station_names: tuple = (
        "cdp",
        "oas",
        "obs",
        "ojp",
        "rme",
        "sap",
        "snb",
        "sod",
        "swa",
        "wfj",
    )

@dataclass(frozen=True)
class paths:

    # ROOT PATH
    root: Path = Path(__file__).resolve().parents[1]
    
    # DATA PATHS
    proc_data: Path = root / "data" / "preprocessed"
    raw_data: Path = root / "data" / "raw"

    # RESULTS PATHS
    models: Path = root / "results" / "models"
    figures: Path = root / "results" / "figures"
    outputs: Path = root / "results" / "outputs"
    simulated_swe: Path = outputs / "simulated_swe"