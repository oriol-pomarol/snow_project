from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class cfg:
    lag: tuple = (1, 14, 140)
    epochs: tuple = (10, 50, 100)
    n_temporal_splits: int = 5
    val_ratio: float = 1/3
    rel_weight: tuple = (0.5, 1, 2)
    station_names: tuple = ("cdp", "oas", "obs", "ojp", "rme",
                            "sap", "snb", "sod", "swa", "wfj")
    trn_stn: tuple = ('cdp', 'rme', 'sod')
    aug_stn: tuple = ('oas', 'obs', 'ojp', 'sap', 'snb', 'swa', 'wfj')
    tst_stn: tuple = aug_stn
    drop_data: float = 0.6
    station_years: tuple = ()

    # Define the modes and the corresponding predictors and target
    def modes():
        return {
    "dir_pred": {"predictors": "^met_", "target": "delta_obs_swe"},
    "err_corr": {"predictors": "^met_", "target": "res_mod_swe"},
    "cro_vars": {"predictors": "^(met_|cro_)", "target": "res_mod_swe"},
    "data_aug": {"predictors": "^met_", "target": "delta_obs_swe"},
    }

    # Set the hyperparameters for each model type
    def hyperparameters(model_type):
        if model_type == 'rf':
            return {
                'max_depth': [None, 10, 20],
                'max_samples': [None, 0.5, 0.8]
                }
        elif model_type == 'nn':
            return {
                'layers': [[2048], [128, 128, 128]],
                'learning_rate': [1e-3, 1e-5],
                'l2_reg': [0, 1e-2, 1e-4]
                }
        elif model_type == 'lstm':
            return {
                'layers': [[512], [128, 64]],
                'learning_rate': [1e-3, 1e-5],
                'l2_reg': [0, 1e-2, 1e-4]
                }   
        else:
            raise ValueError(f"Model type {model_type} not recognized.")
    
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