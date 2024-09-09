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

# @dataclass(frozen=True)
# class paths:
#     data: Path = Path("data")
#     preprocessed: Path = data / "preprocessed"
#     models: Path = Path("models")