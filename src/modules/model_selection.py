import numpy as np
import pandas as pd
import tensorflow as tf
import itertools
from config import cfg, paths
from .model_class import Model
from .auxiliary_functions import (
    load_processed_data,
    temporal_data_split,
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

    # Define the training data in case of a temporal split
    if cfg.temporal_split:
        trn_dfs, _ = temporal_data_split(trn_dfs)

    for mode, mode_vars in cfg.modes().items():
    
        # Obtain the best model for the direct prediction setup
        print(f'Starting {mode} model selection...')
        
        # Take the corresponding predictor and target variables
        X_obs = [df.filter(regex=mode_vars['predictors']) for df in trn_dfs]
        y_obs = [df[[mode_vars['target']]] for df in trn_dfs]
        
        # Take the augmented data if in the corresponding mode
        if mode == 'data_aug':
            X_aug = [df.filter(regex='^met_') for df in aug_dfs]
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

    # Set the hyperparameters for each model type
    rf_hps = {'max_depth': [None, 10, 20],
              'max_samples': [None, 0.5, 0.8]}
    nn_hps = {'layers': [[2048], [128, 128, 128]],
              'learning_rate': [1e-3, 1e-5],
              'l2_reg': [0, 1e-2, 1e-4]}
    lstm_hps = {'layers': [[512], [128, 64]],
                'learning_rate': [1e-3, 1e-5],
                'l2_reg': [0, 1e-2, 1e-4]}
    epochs = [10, 50, 100]

    # Initialize a model for each model type and HP combination
    models = []
    models += initialize_models(mode, 'rf', rf_hps)
    models += initialize_models(mode, 'nn', nn_hps, epochs)
    if cfg.lag > 0:
        models += initialize_models(mode, 'lstm', lstm_hps, epochs)

    # Initialize losses for model validation
    n_splits = 1 if cfg.temporal_split else len(X)
    losses = np.zeros((len(models), n_splits))

    # Iterate over each split
    for s in range(n_splits):

        # Obtain the training, validation and test data
        if cfg.temporal_split:
            X_trn, X_tst, y_trn, y_tst = \
                temporal_split(X, y, s)
        else:
            X_trn, X_tst, y_trn, y_tst = \
                station_split(X, y, s)

        # Add the augmented data if in the corresponding mode
        if mode == 'data_aug':
            X_trn, y_trn, sample_weight = \
                data_aug_split(X_trn, y_trn, X_aug, y_aug)
        else:
            sample_weight = None

        # Iterate through every model
        for m, model in enumerate(models):

            print(f'Split {s+1}/{n_splits}, Model {m+1}/{len(models)}.')

            # Create the model and fit it to the data
            model.create_model(X_trn.shape[1], 0) # Change 0 to the number of crocus variables
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

def initialize_models(mode, model_type, hp_vals_dict, epochs=None):
    
    # Initialize a list of models
    models = []

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

def station_split(X, y, i):

    # Take one station for testing
    X_tst = X[i]
    y_tst = y[i]

    # Concatenate the remaining stations for training
    X_trn = pd.concat([X[j] for j in range(len(X)) if j!=i])
    y_trn = pd.concat([y[j] for j in range(len(y)) if j!=i])

    return X_trn, X_tst, y_trn, y_tst

###############################################################################

def temporal_split(X, y, i):

    # Select the last 20% of each station's data for testing
    X_tst = pd.concat([X[j].tail(int(0.2*len(X[j]))) for j in range(len(X))])
    y_tst = pd.concat([y[j].tail(int(0.2*len(y[j]))) for j in range(len(y))])
    
    # Select the remaining data for training
    X_trn = pd.concat([X[j].head(int(0.8*len(X[j]))) for j in range(len(X))])
    y_trn = pd.concat([y[j].head(int(0.8*len(y[j]))) for j in range(len(y))])

    return X_trn, X_tst, y_trn, y_tst