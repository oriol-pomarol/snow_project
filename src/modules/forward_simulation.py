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
        # Define the number of cross validation splits, with default 1 and no suffix
        n_splits = 1
        suffix = ''

        # If in temporal mode, use the number of temporal splits and corresponding suffix
        if cfg.temporal_split:
            n_splits = cfg.n_temporal_splits
            suffix = 'temp_split'

        # If in data augmentation mode, use the number of test stations
        elif mode == 'data_aug':
            n_splits = len(cfg.tst_stn)
            suffix = 'aug_split'

        # Load the models, also for all temp splits
        model_list = []
        for i in range(n_splits):
            model = Model(mode)
            model.load_hps()
            model.load_model(suffix='_'.join([suffix, str(i)]) if suffix else '')
            model_list.append(model)
        dict_models[mode] = model_list

    # Load the split dates
    if cfg.temporal_split:
        split_dates = pd.read_csv(paths.temp_data / 'split_dates.csv', index_col=[0, 1])

    # Simulate SWE for each station
    for station_idx, (station_name, df_station) in enumerate(dict_dfs.items()):
        print(f"Simulating station {station_idx+1} of {len(cfg.station_names)}.")

        # Drop rows with missing values in the predictors
        predictor_cols = df_station.filter(regex='^(met_|cro_)').columns
        df_stn_clean = df_station.dropna(subset=predictor_cols)

        # Take only a fraction of the data
        df_stn_clean = df_stn_clean.iloc[:int(len(df_stn_clean) * (1 - cfg.drop_data))]

        # If in temporal split mode, append the split index to the dataframe
        if cfg.temporal_split and (station_name in cfg.trn_stn):
            df_stn_clean['temporal_split'] = 0
            split_dates_stn = split_dates.loc[station_name]
            for i, (start_date, end_date) in enumerate(zip(split_dates_stn['tst_start_date'],
                                                           split_dates_stn['tst_end_date'])):
                tst_cond = (df_stn_clean.index >= start_date) & (df_stn_clean.index < end_date)
                df_stn_clean.loc[tst_cond, 'temporal_split'] = i

        # Initialize a vector for the predicted SWE
        pred_swe_arr = np.zeros((df_stn_clean.shape[0], len(dict_models)))

        # Initialize a vector for the zero residual predictions
        pred_class_arr = np.zeros((df_stn_clean.shape[0], len(dict_models)))

        for mode_idx, (mode, mode_vars) in enumerate(cfg.modes().items()):
            print(f"Mode {mode_idx + 1} of {len(dict_models)}.")

            # Get the correct model if in station split mode
            model_list = dict_models[mode]
            model = model_list[0]

            # Check if the conditions are met to use a different model
            dif_model = (not cfg.temporal_split) and (mode == 'data_aug') \
                        and (station_name in cfg.tst_stn)
            
            # Get the index of the test station and select the corresponding model
            if dif_model:
                tst_station_idx = cfg.tst_stn.index(station_name)
                model = model_list[tst_station_idx]
            
            # Get the predictor data for the corresponding mode
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

                # Decide which model to use in temporal split mode
                if cfg.temporal_split and (station_name in cfg.trn_stn):
                    model = model_list[df_stn_clean.loc[df_index, 'temporal_split']]
                    
                # Predict the next SWE value
                pred_y = model.predict(row).ravel()
                 
                # In error correction, subtract the residual from the modelled dSWE
                if cfg.modes()[mode]["target"] == "res_mod_swe":
                    # Use the classifier model to predict if y is zero
                    is_zero = model.predict_classifier(row)
                    pred_class_arr[row_idx, mode_idx] = is_zero
                    if bool(is_zero):
                        pred_y = 0
                    mod_dswe = df_stn_clean.loc[df_index, "delta_mod_swe"]
                    pred_dswe = mod_dswe - pred_y

                # If not predicting the residual, use the predicted value directly
                else:
                    pred_dswe = pred_y

                # Ensure that the predicted SWE is non-negative
                pred_swe = max(pred_swe_arr[row_idx-1, mode_idx] + pred_dswe, 0)

                # Save the predicted SWE and class
                pred_swe_arr[row_idx, mode_idx] = pred_swe
    
        # Save the simulated SWE as a dataframe
        df_swe = pd.DataFrame(pred_swe_arr, index=df_station_X.index, 
                              columns=cfg.modes().keys())
        df_swe.to_csv(paths.temp_data / f'df_{station_name}_pred_swe.csv')

        # Save the simulated class as a dataframe
        df_class = pd.DataFrame(pred_class_arr, index=df_station_X.index, 
                                columns=cfg.modes().keys())
        df_class.to_csv(paths.temp_data / f'df_{station_name}_pred_class.csv')

    return