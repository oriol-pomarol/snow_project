from dataclasses import dataclass
from pathlib import Path

@dataclass()
class cfg:
    lag: tuple = 1
    rel_weights: tuple = (1/2, 1, 2)
    temporal_split: bool = False
    n_temporal_splits: int = 5
    val_ratio: float = 1/3
    drop_data: float = 0.0
    drop_data_expl: float = 0.0
    epochs: tuple = (25, 50, 75, 100)
    station_names: tuple = ("cdp", "oas", "obs", "ojp", "rme",
                            "sap", "snb", "sod", "swa", "wfj")
    trn_stn: tuple = ('cdp', 'rme', 'sod')
    aug_stn: tuple = ('oas', 'obs', 'ojp', 'sap', 'snb', 'swa', 'wfj')
    tst_stn: tuple = aug_stn
    station_years: tuple = ()

    # Define the modes and the corresponding predictors and target
    @staticmethod
    def modes():
        return {
            "dir_pred": "^(met_|obs_swe$)", 
            "post_prc": "^(met_|delta_mod_swe$|obs_swe$|mod_swe$)",
            "cro_vars": "^(met_|cro_|delta_mod_swe$|obs_swe$|mod_swe$)",
            "data_aug": "^(met_|obs_swe$)", 
        }
    
    # Set the hyperparameters to test for each model type
    @staticmethod
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

@dataclass()
class paths:

    # ROOT PATH
    root: Path = Path(__file__).resolve().parents[1]
    
    # DATA PATHS
    proc_data: Path = root / "data" / "processed"
    raw_data: Path = root / "data" / "raw"

    # RESULTS PATHS
    results: Path = root / "results"
    models: Path = results / "models"
    figures: Path = results / "figures"
    outputs: Path = results / "outputs"
    temp: Path = results / "temp"