import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from config import cfg, paths
from .model_class import Model
from .auxiliary_functions import load_processed_data

def forward_simulation():

    # Load the processed data
    dict_dfs = load_processed_data()

    # Clean the dataframes by dropping rows with missing values
    dict_dfs = {station: dropna_aug(df) for station, df in dict_dfs.items()}
    
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
        split_dates = pd.read_csv(paths.temp / 'split_dates.csv', index_col=[0, 1])

    # Simulate SWE for each station
    for station_idx, (station_name, df_station) in enumerate(dict_dfs.items()):
        print(f"Simulating station {station_idx+1} of {len(cfg.station_names)}.")

        # Take only a fraction of the data if specified
        df_station = df_station.iloc[:int(len(df_station) * (1 - cfg.drop_data))]

        # If in temporal split mode, append the split index to the dataframe
        if cfg.temporal_split and (station_name in cfg.trn_stn):
            df_station['temporal_split'] = 0
            split_dates_stn = split_dates.loc[station_name]
            for i, (start_date, end_date) in enumerate(zip(split_dates_stn['tst_start_date'],
                                                           split_dates_stn['tst_end_date'])):
                tst_cond = (df_station.index >= start_date) & (df_station.index < end_date)
                df_station.loc[tst_cond, 'temporal_split'] = i

        # Initialize a vector for the predicted SWE
        pred_swe_arr = np.zeros((df_station.shape[0], len(dict_models)))

        for mode_idx, (mode, predictors) in enumerate(cfg.modes().items()):
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
            df_station_X = df_station.filter(regex=predictors)

            # Get the number of rows in the dataframe
            n_rows = len(df_station_X)

            # Predict each row in the dataframe iteratively
            for row_idx, (df_index, row_serie) in enumerate(df_station_X[:-1].iterrows()):

                # Print progress
                if row_idx % (n_rows // 5) == 0:
                    progress_pct = row_idx * 100 / n_rows
                    print(f"Progress: {progress_pct:.0f}% completed.")

                # Convert the row to a dataframe
                row = row_serie.to_frame().T

                # Add the previous SWE value to the row
                row['obs_swe'] = pred_swe_arr[row_idx, mode_idx]

                # Decide which model to use in temporal split mode
                if cfg.temporal_split and (station_name in cfg.trn_stn):
                    model = model_list[df_station.loc[df_index, 'temporal_split']]
                
                # Predict the next SWE value and ensure it is non-negative
                pred_dswe = model.predict(row).ravel()
                pred_swe = max(pred_swe_arr[row_idx, mode_idx] + pred_dswe, 0)

                # Save the predicted SWE
                pred_swe_arr[row_idx + 1, mode_idx] = pred_swe
    
        # Save the simulated SWE as a dataframe
        df_swe = pd.DataFrame(pred_swe_arr, index=df_station_X.index, 
                              columns=cfg.modes().keys())
        df_swe.to_csv(paths.temp / f'df_{station_name}_pred_swe.csv')


    # Loop over each mode and split to calculate the feature importances
    for mode_idx, (mode, predictors) in enumerate(cfg.modes().items()):
        models = dict_models[mode]

        # Get the training and test data for the SHAP analysis
        trn_dfs = [dict_dfs[station] for station in cfg.trn_stn]
        tst_dfs = [dict_dfs[station] for station in cfg.tst_stn]

        # Take only the same length as the predicted SWE
        trn_dfs = [df.iloc[:len(pred_swe_arr)].copy() for df in trn_dfs]
        tst_dfs = [df.iloc[:len(pred_swe_arr)].copy for df in tst_dfs]

        # Add the previous SWE value to the training and test data from pred_swe_arr
        for i in range(len(trn_dfs)):
            trn_dfs[i].loc[:, 'obs_swe'] = pred_swe_arr[:, mode_idx]
        for i in range(len(tst_dfs)):
            tst_dfs[i].loc[:, 'obs_swe'] = pred_swe_arr[:, mode_idx]

        for s, model in enumerate(models):

            # Take the corresponding training and test data
            X_trn = pd.concat([df.filter(regex=predictors) for df in trn_dfs])
            if cfg.temporal_split:
                y = None
                X_trn, X_tst, _, _ = temporal_test_split(X_trn, y, s)
            else:
                X_tst = tst_dfs[s].filter(regex=predictors)

            # Train the SHAP explainer and get the explanation
            explainer = shap.Explainer(model.predict, X_trn)
            explanation = explainer(X_tst)

            # Plot the absolute SHAP values as bar plots
            plt.figure(figsize=(16,9))
            shap.plots.bar(explanation, max_display=15, show=False)
            plt.savefig(paths.figures / f'val_shap_bar_{mode}_{s}.png', bbox_inches="tight")

            # Plot the SHAP values as violin plots
            plt.figure(figsize=(16,9))
            shap.plots.violin(explanation, max_display=15, show=False)
            plt.savefig(paths.figures / f'val_shap_vio_{mode}_{s}.png', bbox_inches="tight")

            # Save the explanation object to multiple csv files
            save_explanation_to_csv(explanation, mode, s)

    return

###############################################################################

def dropna_aug(df): # Change function name, add to auxiliary_functions.py!!!

    # Drop rows with missing values in the predictors
    predictor_cols = df.filter(regex='^(met_|cro_)').columns
    df_stn_clean = df.dropna(subset=predictor_cols)

    return df_stn_clean

###############################################################################

def temporal_test_split(X, y, split_idx): # Add to auxiliary_functions.py!!!

    # Load the split dates
    df_split_dates = pd.read_csv(paths.temp / 'split_dates.csv', index_col=[0, 1])

    # Initialize lists to store the training and validation data
    X_trn, y_trn, X_tst, y_tst = [], [], [], []

    for i, station in enumerate(cfg.trn_stn):

        # Retrieve the split dates for the current station and split
        tst_start_date, tst_end_date, val_start_date, val_end_date = \
            df_split_dates.loc[(station, split_idx)].values
        
        # Filter the trn and tst data conditions for the current station and split
        trn_cond = (X[i].index < tst_start_date) | \
                   (X[i].index >= tst_end_date)
        tst_cond = (X[i].index >= tst_start_date) & \
                   (X[i].index < tst_end_date)
        # Append the training and test data
        X_trn.append(X[i].loc[trn_cond])
        y_trn.append(y[i].loc[trn_cond])
        X_tst.append(X[i].loc[tst_cond])
        y_tst.append(y[i].loc[tst_cond])          

    return X_trn, X_tst, y_trn, y_tst

###############################################################################

def save_explanation_to_csv(explanation, mode, split_idx): # Add to auxiliary_functions.py!!!

    # Make a folder for the SHAP values
    shap_explanations_path = paths.temp / 'shap_explanations'
    shap_explanations_path.mkdir(exist_ok=True)

    # Get the SHAP values and save them to a csv file
    np.savetxt(shap_explanations_path / f'df_{mode}_shap_{split_idx}.csv',
               explanation.values, delimiter=',')

    # Get the SHAP base values and save them to a csv file
    np.savetxt(shap_explanations_path / f'df_{mode}_base_{split_idx}.csv',
               explanation.base_values, delimiter=',')
    
    # Get the SHAP data and save them to a csv file
    np.savetxt(shap_explanations_path / f'df_{mode}_data_{split_idx}.csv',
               explanation.data, delimiter=',')
    
    return
