from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class cfg:
    lag: int = 14
    temporal_split: bool = True
    n_temporal_splits: int = 5
    val_ratio: float = 0.15
    rel_weight: float = 1
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
    trn_stn: tuple = ('cdp', 'rme', 'sod')
    aug_stn: tuple = ('oas', 'obs', 'ojp', 'sap', 'snb', 'swa')
    tst_stn: tuple = ('wfg',)
    drop_data: float = 0.6

    def modes():
        return {
    "dir_pred": {"predictors": "^met_", "target": "delta_obs_swe"},
    "err_corr": {"predictors": "^(met_|cro_)", "target": "res_mod_swe"},
    "data_aug": {"predictors": "^met_", "target": "delta_obs_swe"},
    }
    
    def __post_init__(self):
        # Check that all stations are in station_names
        assert all(station in self.station_names for station in self.trn_stations), \
        "Some elements in trn_stations are not in station_names"
        assert all(station in self.station_names for station in self.aug_stations), \
        "Some elements in aug_stations are not in station_names"
        assert all(station in self.station_names for station in self.tst_stations), \
        "Some elements in tst_stations are not in station_names"

@dataclass(frozen=True)
class paths:

    # ROOT PATH
    root: Path = Path(__file__).resolve().parents[1]
    
    # DATA PATHS
    proc_data: Path = root / "data" / "processed"
    raw_data: Path = root / "data" / "raw"
    temp_data: Path = root / "data" / "temp"

    # RESULTS PATHS
    models: Path = root / "results" / "models"
    figures: Path = root / "results" / "figures"
    outputs: Path = root / "results" / "outputs"