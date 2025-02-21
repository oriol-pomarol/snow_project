import pandas as pd
from config import cfg, paths
from .model_class import Model
from .auxiliary_functions import (
    load_processed_data,
    replace_obs_dropna,
    integrate_aug_data,
    get_cv_info,
    temporal_test_split
)

def model_training():

    # Load the processed data from all stations
    all_dfs = load_processed_data()
    
    # Store the train, aug and test dataframes and drop NAs
    trn_dfs = [all_dfs[stn].dropna() for stn in cfg.trn_stn]
    aug_dfs = [replace_obs_dropna(all_dfs[stn]) for stn in cfg.aug_stn]
    tst_dfs = [all_dfs[stn].dropna() for stn in cfg.tst_stn]

    # Filter the biased delta SWE values
    trn_dfs = [df.query('delta_obs_swe != -obs_swe') for df in trn_dfs]
    aug_dfs = [df.query('delta_obs_swe != -obs_swe') for df in aug_dfs]
    tst_dfs = [df.query('delta_obs_swe != -obs_swe') for df in tst_dfs]

    # Train the selected models for each mode
    for mode, predictors in cfg.modes().items():
        print(f'Starting {mode} training...')
                    
        # Set the number of cross validation splits to a default of 1
        n_splits, suffix = get_cv_info(mode)

        # Initialize a df to store the predictions
        df_trn = pd.DataFrame(columns=['y_trn', 'y_trn_pred', 'split'])
        df_tst = pd.DataFrame(columns=['y_tst', 'y_tst_pred', 'split'])
        df_aug = pd.DataFrame(columns=['y_aug', 'y_aug_pred', 'split'])

        # Train the model and make predictions in the test set for each split
        for s in range(n_splits):
            print(f'Starting split {s+1} of {n_splits}...')
            
            # Take the corresponding predictor and target variables
            X_obs = [df.filter(regex=predictors) for df in trn_dfs]
            y_obs = [df[['delta_obs_swe']] for df in trn_dfs]

            # Split data into train/test sets and define model suffix for...
            # 1) temporal split 
            if cfg.temporal_split:
                X_trn, X_tst, y_trn, y_tst = temporal_test_split(X_obs, y_obs, s)
                suffix = f'temp_split_{s}'
            # 2) spatial split
            else:
                X_trn, y_trn = X_obs, y_obs
                # 2.1) data augmentation
                if mode == 'data_aug':
                    X_tst = tst_dfs[s].filter(regex=predictors)
                    y_tst = tst_dfs[s][['delta_obs_swe']]
                    suffix = f'aug_split_{s}'

                # 2.2) other setups
                else:
                    X_tst = pd.concat([df.filter(regex=predictors) for df in tst_dfs])
                    y_tst = pd.concat([df[['delta_obs_swe']] for df in tst_dfs])
                    suffix = ''
            
            # Create a new model object and load the hyperparameters
            model = Model(mode)
            model.load_hps()

            # Retrieve the augmented data if in the corresponding mode for...
            X_aug, y_aug, sample_weight = None, None, None
            if mode == 'data_aug':
                # 1) temporal split
                if cfg.temporal_split:
                    X_aug = [df.filter(regex=predictors) for df in aug_dfs]
                    y_aug = [df[['delta_obs_swe']] for df in aug_dfs]
                # 2) spatial split
                else:
                    X_aug = [df.filter(regex=predictors) for i, df in enumerate(aug_dfs) if i != s]
                    y_aug = [df[['delta_obs_swe']] for i, df in enumerate(aug_dfs) if i != s]
                
                # Append the augmented data to the training set
                X_trn, y_trn, sample_weight = \
                    integrate_aug_data(X_trn, y_trn, X_aug, y_aug, model.rel_weight)

            # Count the number of meteo variables and create the model
            n_met_vars = sum([1 for col in X_trn.columns if col.startswith('met_')])
            model.create_model(X_trn.shape[1], n_met_vars)

            # Fit the model to the training data and save it
            model.fit(X_trn, y_trn, save_train_history = True, sample_weight = sample_weight)
            model.save_model(suffix=f"{suffix}_{s}" if suffix else '')

            # Predict the delta SWE for the training and test data
            y_trn_pred = model.predict(X_trn).ravel()
            y_tst_pred = model.predict(X_tst).ravel()

            # If in data augmentation, predict delta SWE for the augmented data
            if mode == 'data_aug':
                X_aug_df = pd.concat(X_aug)
                y_aug_pred = model.predict(X_aug_df).ravel()
                y_aug = pd.concat(y_aug).values.ravel()

            # Concatenate the observed values and convert to 1D numpy array
            y_trn = y_trn.values.ravel()
            y_tst = y_tst.values.ravel()

            # Append the predictions to the dataframes
            dict_trn = {'y_trn': y_trn, 'y_trn_pred': y_trn_pred, 'split': s}
            df_trn = pd.concat([df_trn, pd.DataFrame(dict_trn)], ignore_index=True)
            dict_tst = {'y_tst': y_tst, 'y_tst_pred': y_tst_pred, 'split': s}
            df_tst = pd.concat([df_tst, pd.DataFrame(dict_tst)], ignore_index=True)
            if mode == 'data_aug':
                dict_aug = {'y_aug': y_aug, 'y_aug_pred': y_aug_pred, 'split': s}
                df_aug = pd.concat([df_aug, pd.DataFrame(dict_aug)], ignore_index=True)

        # Save the dataframes
        df_trn.to_csv(paths.temp / f'pred_vs_true_{mode}_trn.csv', index=False)
        df_tst.to_csv(paths.temp / f'pred_vs_true_{mode}_tst.csv', index=False)
        if mode == 'data_aug':
            df_aug.to_csv(paths.temp / f'pred_vs_true_{mode}_aug.csv', index=False)
        
        print(f'{mode} trained successfully...')

    return