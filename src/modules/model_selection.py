import numpy as np
import pandas as pd
import tensorflow as tf
import itertools
from sklearn.model_selection import train_test_split
from config import cfg, paths
from model_class import Model
from auxiliary_functions import load_processed_data, temporal_data_split

def model_selection():

    # Load the processed data from all stations
    all_dfs = load_processed_data()
    
    # Store the training and augmentation dataframes
    trn_dfs = [all_dfs[station] for station in cfg.trn_stn]
    aug_dfs = [all_dfs[station] for station in cfg.aug_stn]

    # Filter the biased delta SWE values
    trn_dfs = [df.query('delta_obs_swe != -obs_swe') for df in trn_dfs]
    aug_dfs = [df.query('delta_mod_swe != -mod_swe') for df in aug_dfs]
    
    # Set a random seed for tensorflow
    tf.random.set_seed(10)

    # Define the training data in case of a temporal split
    if cfg.temporal_split:
        trn_dfs, _ = temporal_data_split(trn_dfs)
    
    # Obtain the best model for the direct prediction setup
    print('Starting direct prediction training...')

    # Take meteorological data as input and observed delta SWE as output
    X_obs = [df.filter(regex='^met_') for df in trn_dfs]
    y_obs = [df[['delta_obs_swe']] for df in trn_dfs]

    # Obtain the best model and save its hyperparameters
    model_dp = select_model(X=X_obs, y=y_obs, mode = 'dir_pred')
    model_dp.save_hps()
    print('Direct prediction trained successfully...')

    # Obtain the best model for the error correction setup
    print('Starting error correction training...')

    # Take meteorological and crocus data as input and SWE residuals as output
    X_obs = [df.filter(regex='^(met_|cro_)') for df in trn_dfs]
    y_obs = [df[['res_mod_swe']] for df in trn_dfs]

    # Obtain the best model and save its hyperparameters
    model_ec = select_model(X=X_obs, y=y_obs, mode = 'err_corr')
    model_ec.save_hps()
    print('Error correction trained successfully...')

    # Obtain the best model for the data augmentation setup
    print('Starting data augmentation training...')

    # Take meteorological data as input and observed delta SWE as output
    X_obs = [df.filter(regex='^met_') for df in trn_dfs]
    y_obs = [df[['delta_obs_swe']] for df in trn_dfs]

    # Take meteorological data as input and modelled delta SWE as output
    X_aug = [df.filter(regex='^met_') for df in aug_dfs]
    y_aug = [df[['delta_mod_swe']] for df in aug_dfs]

    # Obtain the best model and save its hyperparameters
    model_da = select_model(X=X_obs, y=y_obs, X_aug=X_aug,
                            y_aug=y_aug, mode = 'data_aug')
    model_da.save_hps()
    print('Data augmentation trained successfully...')

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

    # Initialize a model for each model type and HP combination
    models = []
    models += initialize_models(mode, 'rf', rf_hps)
    models += initialize_models(mode, 'nn', nn_hps)
    if cfg.lag > 0:
        models += initialize_models(mode, 'lstm', lstm_hps)

    # Initialize losses for model validation
    n_splits = 1 if cfg.temporal_split else len(X)
    losses = np.zeros((len(models), n_splits))

    # Iterate over each split
    for s in len(n_splits):

        # Obtain the training, validation and test data
        if cfg.temporal_split:
            X_trn, X_val, X_tst, y_trn, y_val, y_tst = \
                temporal_split(X, y, s)
        else:
            X_trn, X_val, X_tst, y_trn, y_val, y_tst = \
                station_split(X, y, s)

        # Add the augmented data if in the corresponding mode
        if mode == 'data_aug':
            X_trn, y_trn, X_val_aug, y_val_aug, sample_weight = \
                data_aug_split(X_trn, y_trn, X_aug, y_aug)
        else:
            X_val_aug, y_val_aug, sample_weight = None, None, None

        # Iterate through every model
        for m, model in enumerate(models):

            print(f'Split {s*1}/{n_splits}, Model {m+1}/{len(models)}.')

            # Create the model and fit it to the data
            model.create_model(X_trn.shape[1])
            model.fit(X_trn, y_trn, X_val, y_val,
                      X_val_aug, y_val_aug, sample_weight)
            
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

def initialize_models(mode, model_type, hp_vals_dict):
    
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
        model = Model(model_type, mode, hp_combination)

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

    # Take a random subset for validation
    X_trn, X_val, y_trn, y_val = \
        train_test_split(X_trn, y_trn, test_size=0.1, random_state=10)

    return X_trn, X_val, X_tst, y_trn, y_val, y_tst

###############################################################################

def temporal_split(X, y, i):

    # Select the last 20% of each station's data for testing
    X_tst = pd.concat([X[j].tail(int(0.2*len(X[j]))) for j in range(len(X))])
    y_tst = pd.concat([y[j].tail(int(0.2*len(y[j]))) for j in range(len(y))])

    # Select the 10% before the test data for validation
    X_val = pd.concat([X[j].tail(int(0.3*len(X[j]))).head(int(0.1*len(X[j])))
                       for j in range(len(X))])
    y_val = pd.concat([y[j].tail(int(0.3*len(y[j]))).head(int(0.1*len(y[j])))
                       for j in range(len(y))])
    
    # Select the remaining data for training
    X_trn = pd.concat([X[j].head(int(0.7*len(X[j]))) for j in range(len(X))])
    y_trn = pd.concat([y[j].head(int(0.7*len(y[j]))) for j in range(len(y))])

    return X_trn, X_val, X_tst, y_trn, y_val, y_tst

###############################################################################

def data_aug_split(X_trn, y_trn, X_aug, y_aug):

    # Take a random subset for validation
    X_trn_aug, X_val_aug, y_trn_aug, y_val_aug = \
        train_test_split(pd.concat(X_aug), pd.concat(y_aug),
                         test_size=0.1, random_state=10)
    
    # Calculate the training weights of the modelled data
    weight_aug = cfg.rel_weight * len(X_trn) / len(X_trn_aug)
    sample_weight = np.concatenate((np.ones(len(X_trn)), 
                                    np.full(len(X_trn_aug), weight_aug)))
    
    # Concatenate the observed and augmented datasets
    X_trn = pd.concat([X_trn, X_trn_aug])
    y_trn = pd.concat([y_trn, y_trn_aug])

    return X_trn, y_trn, X_val_aug, y_val_aug, sample_weight