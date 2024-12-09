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
        split_dates = pd.read_csv(paths.temp_data / 'split_dates.csv', index_col=[0, 1])

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
            df_station_X = df_station.filter(regex=mode_vars['predictors'])

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
                    model = model_list[df_station.loc[df_index, 'temporal_split']]
                
                # Predict the next SWE value
                pred_dswe = model.predict(row).ravel()
                 
                # In error correction, subtract the residual from the modelled dSWE
                if cfg.modes()[mode]["target"] == "res_mod_swe":
                    mod_dswe = df_station.loc[df_index, "delta_mod_swe"]
                    pred_dswe = mod_dswe - pred_dswe

                # Ensure that the predicted SWE is non-negative
                pred_swe = max(pred_swe_arr[row_idx-1, mode_idx] + pred_dswe, 0)

                # Save the predicted SWE
                pred_swe_arr[row_idx, mode_idx] = pred_swe
    
        # Save the simulated SWE as a dataframe
        df_swe = pd.DataFrame(pred_swe_arr, index=df_station_X.index, 
                              columns=cfg.modes().keys())
        df_swe.to_csv(paths.temp_data / f'df_{station_name}_pred_swe.csv')

    # Gather the data from the training and test stations used for forward simulation
    trn_dfs = [dict_dfs[station] for station in cfg.trn_stn]
    tst_dfs = [dict_dfs[station] for station in cfg.tst_stn]


    # Loop over each mode and split to calculate the feature importances
    for mode, mode_vars in cfg.modes().items():
        models = dict_models[mode]
        for s, model in enumerate(models):

            # Take the corresponding data
            if cfg.temporal_split:
                X = pd.concatenate([dict_dfs[station] for station in cfg.trn_stn])
                y = None # ADD Y HERE
                X_trn, X_tst, y_trn, y_tst = temporal_test_split(X, y, s)
            else:
                X_trn = pd.concatenate(trn_dfs)
                X_tst = pd.concatenate(tst_dfs)

            # SHAP IMPORTANCE
            explainer = shap.Explainer(model.predict, X_trn)
            explanation = explainer(X_tst)

            plt.figure(figsize=(16,9))
            shap.plots.bar(explanation, max_display=15, show=False)
            plt.savefig(paths.figures / f'val_shap_bar_{mode}_{s}.png', bbox_inches="tight")

            plt.figure(figsize=(16,9))
            shap.plots.violin(explanation, max_display=15, show=False)
            plt.savefig(paths.figures / f'val_shap_vio_{mode}_{s}.png', bbox_inches="tight")

            ## PERMUTATION IMPORTANCE
            # train_importances = permutation_importance(model, X_train, y_train, scoring= 'neg_mean_squared_error',
            #                                            n_repeats = 50, random_state=10).importances
            # val_importances = permutation_importance(model, X_val, y_val, scoring= 'neg_mean_squared_error',
            #                                          n_repeats = 50, random_state=10).importances
            # np.savetxt(os.path.join('results', f'train_importances_{mode}.txt'),
            #            train_importances, delimiter=',', header='Feature Importance (train)')
            # np.savetxt(os.path.join('results', f'val_importances_{mode}.txt'),
            #            val_importances, delimiter=',', header='Feature Importance (val)')

            # # Calculate the mean importance values and sort the indices
            # sorted_indices = train_importances.mean(axis=1).argsort()

            # # Sort both the columns and importances based on sorted indices
            # sorted_columns = X_train.columns[sorted_indices]
            # sorted_importances = train_importances[sorted_indices]

            # # Plot the whisker plots (train)
            # plt.figure(figsize=(10, 6))
            # plt.boxplot(sorted_importances.T, vert=False)
            # plt.yticks(range(1, len(sorted_columns) + 1), sorted_columns)
            # plt.xlabel('Importance')
            # plt.title('Feature Importances (Permutation)')
            # plt.tight_layout()
            # plt.savefig(os.path.join('results',f'train_importances_{mode}.png'))

            # # Calculate the mean importance values and sort the indices
            # sorted_indices = val_importances.mean(axis=1).argsort()

            # # Sort both the columns and importances based on sorted indices
            # sorted_columns = X_val.columns[sorted_indices]
            # sorted_importances = val_importances[sorted_indices]

            # # Plot the whisker plots (val)
            # plt.figure(figsize=(10, 6))
            # plt.boxplot(sorted_importances.T, vert=False)
            # plt.yticks(range(1, len(sorted_columns) + 1), sorted_columns)
            # plt.xlabel('Importance')
            # plt.title('Feature Importances (Permutation)')
            # plt.tight_layout()
            # plt.savefig(os.path.join('results',f'val_importances_{mode}.png'))

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
    df_split_dates = pd.read_csv(paths.temp_data / 'split_dates.csv', index_col=[0, 1])

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
