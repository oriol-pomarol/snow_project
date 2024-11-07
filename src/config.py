from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class cfg:
    lag: tuple = 1
    rel_weight: tuple = 1
    temporal_split: bool = False
    n_temporal_splits: int = 5
    val_ratio: float = 1/3
    drop_data: float = 0.0
    epochs: tuple = (25, 50, 75, 100)
    station_names: tuple = ("cdp", "oas", "obs", "ojp", "rme",
                            "sap", "snb", "sod", "swa", "wfj")
    trn_stn: tuple = ('cdp', 'rme', 'sod')
    aug_stn: tuple = ('oas', 'obs', 'ojp', 'sap', 'snb', 'swa', 'wfj')
    tst_stn: tuple = aug_stn
    station_years: tuple = ()

    # Define the modes and the corresponding predictors and target
    def modes():
        return {
            "dir_pred": {
                "predictors": "^met_", 
                "target": "delta_obs_swe"
            },
            "err_corr": {
                "predictors": "^met_", 
                "target": "res_mod_swe"
            },
            "cro_vars": {
                "predictors": "^(met_|cro_)", 
                "target": "res_mod_swe"
            },
            "data_aug": {
                "predictors": "^met_", 
                "target": "delta_obs_swe"
            },
        }
    
    # Set the hyperparameters to test for each model type
    def hyperparameters():
        return {
            'rf': {
                'max_depth': [None, 10, 20],
                'max_samples': [None, 0.5, 0.8]
            },
            'nn': {
                'layers': [[2048], [128, 128, 128]],
                'learning_rate': [1e-3, 1e-5],
                'l2_reg': [0, 1e-2, 1e-4]
            },
            'lstm': {
                'layers': [[512], [128, 64]],
                'learning_rate': [1e-3, 1e-5],
                'l2_reg': [0, 1e-2, 1e-4]
            },
        }

@dataclass(frozen=True)
class paths:

    # ROOT PATH
    root: Path = Path(__file__).resolve().parents[1]
    
    # DATA PATHS
    proc_data: Path = root / "data" / "processed"
    raw_data: Path = root / "data" / "raw"
    temp_data: Path = root / "data" / "temp"

    # RESULTS PATHS
    results: Path = root / "results"
    models: Path = results / "models"
    figures: Path = results / "figures"
    outputs: Path = results / "outputs"