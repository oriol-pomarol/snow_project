import numpy as np
import pandas as pd
from config import cfg, paths
from .model_class import Model
from .auxiliary_functions import load_processed_data

def forward_simulation():

    # Load the processed data
    dict_dfs = load_processed_data()
    
    # Load the models for every mode
    dict_models = {}
    for mode in cfg.modes().keys():
        # Load the models, also for all data_aug splits
        model_list = []
        for i in range(len(cfg.aug_stn)) if mode == 'data_aug' else range(1):
            model = Model(mode)
            model.load_hps()
            model.load_model(suffix=f"aug_split_{i}" if mode == 'data_aug' else '')
            model_list.append(model)
        dict_models[mode] = model_list

    # Simulate SWE for each station
    for station_idx, (station_name, df_station) in enumerate(dict_dfs.items()):
        print(f"Simulating station {station_idx+1} of {len(cfg.station_names)}.")

        # Drop rows with missing values in the predictors
        predictor_cols = df_station.filter(regex='^(met_|cro_)').columns
        df_stn_clean = df_station.dropna(subset=predictor_cols)

        # Initialize a vector for the predicted SWE
        pred_swe_arr = np.zeros((df_stn_clean.shape[0], len(dict_models)))

        for mode_idx, (mode, mode_vars) in enumerate(cfg.modes().items()):
            print(f"Mode {mode_idx + 1} of {len(dict_models)}.")

            # Get the model and the predictor data
            model_list = dict_models[mode]
            if (mode == 'data_aug') and (station_name in cfg.tst_stn):
                # Retrieve index of the station in the test stations
                tst_station_idx = cfg.tst_stn.index(station_name)
                # Select the model corresponding to the test station
                model = model_list[tst_station_idx]
            else:
                # Select the first model in the list
                model = model_list[0]
            df_station_X = df_stn_clean.filter(regex=mode_vars['predictors'])

            # Get the number of rows in the dataframe
            n_rows = len(df_station_X)

            # Predict each row in the dataframe iteratively
            for row_idx, (df_index, row_serie) in enumerate(df_station_X.iterrows()):
                row = row_serie.to_frame().T

                # Print progress
                if row_idx % (n_rows // 5) == 0:
                    progress_pct = row_idx * 100 / n_rows
                    print(f"Progress: {progress_pct:.0f}% completed.")
                
                # Predict the next SWE value
                pred_dswe = model.predict(row).ravel()
                 
                # In error correction, subtract the residual from the modelled dSWE
                if mode == "err_corr":
                    mod_dswe = df_stn_clean.loc[df_index, "delta_mod_swe"]
                    pred_dswe = mod_dswe - pred_dswe

                # Ensure that the predicted SWE is non-negative
                pred_swe = max(pred_swe_arr[row_idx-1, mode_idx] + pred_dswe, 0)

                # Save the predicted SWE
                pred_swe_arr[row_idx, mode_idx] = pred_swe
    
        # Save the simulated SWE as a dataframe
        df_swe = pd.DataFrame(pred_swe_arr, index=df_station_X.index, 
                              columns=cfg.modes().keys())
        df_swe.to_csv(paths.temp_data / f'df_{station_name}_pred_swe.csv')

    return