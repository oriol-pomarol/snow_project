import numpy as np
import pandas as pd
import tensorflow as tf
import itertools
from config import cfg, paths
from .model_class import Model
from .auxiliary_functions import (
    load_processed_data,
    find_temporal_split_dates,
    data_aug_split,
)

def model_selection():

    # Load the processed data from all stations
    all_dfs = load_processed_data()
    
    # Store the training and augmentation dataframes and drop NAs
    trn_dfs = [all_dfs[stn].dropna() for stn in cfg.trn_stn]
    ignore_cols = ["delta_obs_swe", "obs_swe", "res_mod_swe"]
    dropna_cols = [col for col in trn_dfs[0].columns if col not in ignore_cols]
    aug_dfs = [all_dfs[stn].dropna(subset=dropna_cols) for stn in cfg.aug_stn]

    # Filter the biased delta SWE values
    trn_dfs = [df.query('delta_obs_swe != -obs_swe') for df in trn_dfs]
    aug_dfs = [df.query('delta_mod_swe != -mod_swe') for df in aug_dfs]
    
    # Set a random seed for tensorflow
    tf.random.set_seed(10)

    # Find the dates for the temporal split
    if cfg.temporal_split:
        find_temporal_split_dates(trn_dfs)

    for mode, mode_vars in cfg.modes().items():
    
        # Obtain the best model for the direct prediction setup
        print(f'Starting {mode} model selection...')
        
        # Take the corresponding predictor and target variables
        X_obs = [df.filter(regex=mode_vars['predictors']) for df in trn_dfs]
        y_obs = [df[[mode_vars['target']]] for df in trn_dfs]
        
        # Take the augmented data if in the corresponding mode
        if mode == 'data_aug':
            X_aug = [df.filter(regex='^met_' + '|^mod_swe$') for df in aug_dfs]
            y_aug = [df[['delta_mod_swe']] for df in aug_dfs]
        else:
            X_aug, y_aug = None, None

        # Obtain the best model and save its hyperparameters
        model = select_model(X = X_obs, y = y_obs, X_aug = X_aug,
                            y_aug = y_aug, mode = mode)
        model.save_hps()
        print(f'{mode} model selected successfully...')

    return

###############################################################################
# SELECT MODEL FUNCTION
###############################################################################

def select_model(X, y, X_aug=None, y_aug=None, mode='dir_pred'):    

    # Initialize a model for each model type and HP combination
    models = initialize_models(mode)

    # Initialize losses for model validation
    n_splits = cfg.n_temporal_splits if cfg.temporal_split else len(X)
    losses = np.zeros((len(models), n_splits))

    # Iterate over each split
    for s in range(n_splits):

        # Obtain the training, validation and test data
        if cfg.temporal_split:
            X_trn, X_tst, y_trn, y_tst = \
                temporal_validation_split(X, y, s)
        else:
            X_trn, X_tst, y_trn, y_tst = \
                station_validation_split(X, y, s)

        # Add the augmented data if in the corresponding mode
        if mode == 'data_aug':
            X_trn, y_trn, sample_weight = \
                data_aug_split(X_trn, y_trn, X_aug, y_aug)
        else:
            sample_weight = None

        # Iterate through every model
        for m, model in enumerate(models):

            print(f'Split {s+1}/{n_splits}, Model {m+1}/{len(models)}.')

            # Count the number of meteo variables
            n_met_vars = sum([1 for col in X_trn.columns if col.startswith('met_')])

            # Create the model and fit it to the data
            model.create_model(X_trn.shape[1], n_met_vars)
            model.fit(X_trn, y_trn, sample_weight=sample_weight)
            
            # Test the model on the validation data and store the loss
            loss = model.test(X=X_tst, y=y_tst)
            losses[m, s] = loss

    # Select the best model
    mean_loss = np.mean(losses, axis=1)
    best_model = models[np.argmin(mean_loss)]

    # Save the model hyperparameters and their losses as a csv
    model_names = [str(model) for model in models]
    if losses.shape[1] == 1:
        df_losses = pd.DataFrame({'MSE': losses[:, 0], 'HP': model_names})
    else:
        data = {f'MSE (Split {i+1})': losses[:, i] for i in range(losses.shape[1])}
        data.update({'MSE (mean)': mean_loss, 'HP': model_names})
        df_losses = pd.DataFrame(data)
    df_losses.set_index('HP', inplace=True)
    df_losses.to_csv(paths.outputs / f'model_losses_{mode}.csv')

    return best_model

###############################################################################

def initialize_models(mode):
    
    # Initialize a list of models
    models = []

    # Loop over each model type
    for model_type in ['rf', 'nn', 'lstm']:

        # Get the hyperparameters for the model type
        hp_vals_dict = cfg.hyperparameters(model_type)

        # Set the epochs for the model
        if model_type == 'rf':
            epochs = None
        else:
            epochs = cfg.epochs

        # Create a list of HP names and all possible HP combinations
        hp_names = list(hp_vals_dict.keys())
        hp_vals = itertools.product(*hp_vals_dict.values())

        # Iterate over each HP combination
        for hp_val_combination in hp_vals:

            # Create a dictionary with each HP combination and names
            hp_combination = dict(zip(hp_names, hp_val_combination))

            # Create a model with the HP combination
            model = Model(mode)
            model.set_hps(model_type, hp_combination, epochs)

            # Append the model to the list
            models.append(model)

    return models

###############################################################################

def station_validation_split(X, y, i):

    # Take one station for testing
    X_tst = X[i]
    y_tst = y[i]

    # Concatenate the remaining stations for training
    X_trn = pd.concat([X[j] for j in range(len(X)) if j!=i])
    y_trn = pd.concat([y[j] for j in range(len(y)) if j!=i])

    return X_trn, X_tst, y_trn, y_tst

###############################################################################

def temporal_validation_split(X, y, split_idx):

    # Load the split dates
    df_split_dates = pd.read_csv(paths.temp_data / 'split_dates.csv', index_col=[0, 1])

    # Initialize lists to store the training and validation data
    X_trn, y_trn, X_val, y_val = [], [], [], []

    for i, station in enumerate(cfg.trn_stn):

        # Retrieve the split dates for the current station and split
        tst_start_date, tst_end_date, val_start_date, val_end_date = \
            df_split_dates.loc[(station, split_idx)].values
        
        # Filter the trn and val data conditions for the current station and split
        trn_cond = ((X[i].index < tst_start_date) | \
                    (X[i].index >= tst_end_date)) & \
                   ((X[i].index < val_start_date) | \
                    (X[i].index >= val_end_date))
     
        val_cond = (X[i].index >= val_start_date) & \
                   (X[i].index < val_end_date)

        # Append the training and validation data
        X_trn.append(X[i].loc[trn_cond])
        y_trn.append(y[i].loc[trn_cond])
        X_val.append(X[i].loc[val_cond])
        y_val.append(y[i].loc[val_cond])        

    # Concatenate the training and validation data
    X_trn, y_trn = pd.concat(X_trn), pd.concat(y_trn)
    X_val, y_val = pd.concat(X_val), pd.concat(y_val)

    return X_trn, X_val, y_trn, y_val