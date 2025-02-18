import numpy as np
import pandas as pd
import itertools
from config import cfg, paths
from .model_class import Model
from .auxiliary_functions import (
    load_processed_data,
    find_temporal_split_dates,
    integrate_aug_data,
    replace_obs_dropna,
    temporal_validation_split,
    station_validation_split,
)

def model_selection():

    # Load the processed data from all stations
    all_dfs = load_processed_data()
    
    # Store the training and augmentation dataframes and drop NAs
    trn_dfs = [all_dfs[stn].dropna() for stn in cfg.trn_stn]
    aug_dfs = [replace_obs_dropna(all_dfs[stn]) for stn in cfg.aug_stn]

    # Filter the biased delta SWE values
    trn_dfs = [df.query('delta_obs_swe != -obs_swe') for df in trn_dfs]
    aug_dfs = [df.query('delta_obs_swe != -obs_swe') for df in aug_dfs]
    
    # Find the dates for the temporal split
    if cfg.temporal_split:
        find_temporal_split_dates(trn_dfs)

    # Select the best model and HP for each mode
    for mode, predictors in cfg.modes().items():

        print(f'Starting {mode} model selection...')

        # Filter the corresponding predictor and target variables
        X_obs = [df.filter(regex=predictors) for df in trn_dfs]
        y_obs = [df[['delta_obs_swe']] for df in trn_dfs]
        
        # Filter the augmented data if in the corresponding mode
        if mode == 'data_aug':
            X_aug = [df.filter(regex=predictors) for df in aug_dfs]
            y_aug = [df[['delta_obs_swe']] for df in aug_dfs]
        else:
            X_aug, y_aug = None, None

        # Initialize a model for each model type and HP combination
        models = initialize_models(mode)

        # Get the losses for each model and split with rel_weight = 1
        losses = get_losses(X_obs, y_obs, X_aug, y_aug,
                            models, mode)

        # Select the best model based on the losses
        mean_losses = np.mean(losses, axis=1)
        best_model = models[np.argmin(mean_losses)]

        # Set epochs to the median of best epochs for all splits, if not a RF
        if best_model.model_type != 'rf':
            best_model.epochs = [int(np.median(best_model.best_epochs)),]
            print(f'Saved model epochs at: {best_model.epochs[0]}')

        # Tune rel_weight, if specified and in data_aug mode
        if mode == 'data_aug' and cfg.rel_weights:

            # Initialize a list of the best model with different rel_weights
            models_aug = [best_model.copy() for _ in cfg.rel_weights]
            for model, rel_weight in zip(models_aug, cfg.rel_weights):
                model.rel_weight = rel_weight

            # Get the losses from each model with different rel_weights
            losses_aug = get_losses(X_obs, y_obs, X_aug, y_aug,
                                    models_aug, mode)

            # Find the best relative weight and set it to the best model
            mean_losses_aug = np.mean(losses_aug, axis=1)
            best_rel_weight = cfg.rel_weights[np.argmin(mean_losses_aug)]
            best_model.rel_weight = best_rel_weight

            # Concatenate the augmented and mean losses and models
            losses = np.concatenate((losses, losses_aug))
            mean_losses = np.concatenate((mean_losses, mean_losses_aug))
            models += models_aug

        # Save the hyperparameters and losses to a csv
        best_model = save_hp_losses(models, losses, mean_losses, mode)

        # Save the best model hyperparameters to a json
        best_model.save_hps()

        print(f'{mode} model selected successfully...')

    return

###############################################################################
# MODEL SELECTION FUNCTIONS
###############################################################################

def initialize_models(mode):
    
    # Initialize a list of models
    models = []

    # Loop over each model type and hyperparameter combination
    for model_type, hp_vals_dict in cfg.hyperparameters().items():

        # If lag is 0, skip LSTM models
        if cfg.lag == 0 and model_type == 'lstm':
            continue

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

            # If in data_aug mode, set the rel_weight to 1
            if mode == 'data_aug':
                model.rel_weight = 1

            # Append the model to the list
            models.append(model)

    return models

###############################################################################

def get_losses(X, y, X_aug, y_aug, models, mode):

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

        # Iterate through every model
        for m, model in enumerate(models):

            print(f'Split {s+1}/{n_splits}, Model {m+1}/{len(models)}.')

            # Add the augmented data if in the corresponding mode
            if mode == 'data_aug':
                X_trn, y_trn, sample_weight = \
                    integrate_aug_data(X_trn, y_trn, X_aug, y_aug, model.rel_weight)
            else:
                sample_weight = None

            # Count the number of meteo variables
            n_met_vars = sum([1 for col in X_trn.columns if col.startswith('met_')])

            # Create the model and fit it to the data
            model.create_model(X_trn.shape[1], n_met_vars)
            model.fit(X_trn, y_trn, sample_weight=sample_weight)
            
            # Test the model on the validation data and store the loss
            loss = model.test(X=X_tst, y=y_tst)
            losses[m, s] = loss
    return losses

###############################################################################

def save_hp_losses(models, losses, mean_losses, mode):    

    # Save the model hyperparameters and their losses as a csv
    model_names = [str(model) for model in models]

    # Save losses as a single column, if only one split
    if losses.shape[1] == 1:
        df_losses = pd.DataFrame({'MSE': losses[:, 0], 'HP': model_names})
    
    # Save losses as multiple columns and compute the mean, if multiple splits
    else:
        data = {f'MSE (Split {i+1})': losses[:, i] for i in range(losses.shape[1])}
        data.update({'MSE (mean)': mean_losses, 'HP': model_names})
        df_losses = pd.DataFrame(data)

    # Set the HP column as the index and save the dataframe to a csv
    df_losses.set_index('HP', inplace=True)
    df_losses.to_csv(paths.outputs / f'model_losses_{mode}.csv')

    return