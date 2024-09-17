import numpy as np
import pandas as pd
from config import cfg, paths
from .model_class import Model
from .auxiliary_functions import load_processed_data

def forward_simulation():

    # Load the processed data
    dict_dfs = load_processed_data()
    
    # Load the models
    list_models = []
    for mode in cfg.modes().keys():
        model = Model(mode)
        model.load_hps()
        model.load_model()
        list_models.append(model)

    # Simulate SWE for each station
    for station_idx, (station_name, df_station) in enumerate(dict_dfs.items()):
        print(f"Simulating station {station_idx+1} of {len(cfg.station_names)}.")

        # Initialize a vector for the predicted SWE
        drop_cols_X = ["obs_swe", "delta_obs_swe", "mod_swe"]
        df_station_X = df_station.drop(drop_cols_X, axis=1).dropna()
        pred_swe_arr = np.zeros((len(list_models), len(df_station_X)))

        for model_idx, model in enumerate(list_models):
            print(f"Mode {model_idx + 1} of {len(list_models)}.")

            for row_idx, row_serie in df_station_X.iterrows():
                row = row_serie.to_frame().T
                # Print progress
                if row_idx % (len(df_station_X) // 5) == 0:
                    progress_pct = row_idx * 100 / len(df_station_X)
                    print(f"Progress: {progress_pct:.0f}% completed.")
                
                # Predict the next SWE value
                pred_dswe = model.predict(row).ravel()
                if model_idx == 1:
                    pred_dswe = row["delta_mod_swe"].values - pred_dswe
                pred_swe = max(pred_swe_arr[model_idx,row_idx-1] + pred_dswe, 0)
                pred_swe_arr[model_idx, row_idx] = pred_swe
    
        # Save the simulated SWE as a dataframe
        df_swe = pd.DataFrame(pred_swe_arr.T, index=df_station_X.index, 
                              columns=cfg.modes().keys())
        df_swe.to_csv(paths.temp_data / f'df_{station_name}_sim_swe.csv')

    return