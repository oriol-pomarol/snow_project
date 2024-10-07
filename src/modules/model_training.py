import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import r2_score
from config import cfg, paths
from .model_class import Model
from .auxiliary_functions import (
    load_processed_data,
    data_aug_split
)

def model_training():

    # Load the processed data from all stations
    all_dfs = load_processed_data()
    
    # Store the training and augmentation dataframes and drop NAs
    trn_dfs = [all_dfs[station].dropna() for station in cfg.trn_stn]
    aug_dfs = [all_dfs[station].dropna() for station in cfg.aug_stn]

    # Filter the biased delta SWE values
    trn_dfs = [df.query('delta_obs_swe != -obs_swe') for df in trn_dfs]
    aug_dfs = [df.query('delta_mod_swe != -mod_swe') for df in aug_dfs]
    
    # Set a random seed for tensorflow
    tf.random.set_seed(10)

    for mode, mode_vars in cfg.modes().items():
        print(f'Starting {mode} training...')
                    
        # Define the number of splits for training the model
        n_splits = cfg.n_temporal_splits if cfg.temporal_split else 1

        for s in range(n_splits):
            print(f'Starting split {s+1} of {n_splits}...')
            
            # Take the corresponding predictor and target variables
            X_obs = [df.filter(regex=mode_vars['predictors']) for df in trn_dfs]
            y_obs = [df[[mode_vars['target']]] for df in trn_dfs]

            # Split the data into training and test sets
            if cfg.temporal_split:
                X_trn, X_tst, y_trn, y_tst = temporal_test_split(X_obs, y_obs, s)
        
            else:
                X_trn, y_trn = X_obs, y_obs
                tst_dfs = [all_dfs[station].dropna() for station in cfg.tst_stn]
                X_tst = [df.filter(regex=mode_vars['predictors']) for df in tst_dfs]
                y_tst = [df[[mode_vars['target']]] for df in tst_dfs]

            # Take the augmented data if in the corresponding mode
            if mode == 'data_aug':
                X_aug = [df.filter(regex='^met_') for df in aug_dfs]
                y_aug = [df[['delta_mod_swe']] for df in aug_dfs]
            else:
                X_aug, y_aug = None, None
                
            # Train the best model and save it
            model = train_model(X_obs, y_obs, X_aug, y_aug, mode = mode)
            suffix = f'temp_split_{s}' if cfg.temporal_split else ''
            model.save_model(suffix=suffix)

            # Take the test data for direct prediction
            X_tst = [df.filter(regex=mode_vars['predictors']) for df in tst_dfs]
            y_tst = [df[[mode_vars['target']]] for df in tst_dfs]

            # Predict the delta SWE for the training and test data
            y_train_pred = model.predict(pd.concat(X_obs)).ravel()
            y_tst_pred = model.predict(pd.concat(X_tst)).ravel()

            # If in data augmentation, predict delta SWE for the augmented data
            if mode == 'data_aug':
                y_aug_pred = model.predict(pd.concat(X_aug)).ravel()
                y_aug = pd.concat(y_aug).values.ravel()
            else:
                y_aug_pred = None

            # Concatenate the observed values and convert to 1D numpy array
            y_train = pd.concat(y_obs).values.ravel()
            y_tst = pd.concat(y_tst).values.ravel()

            # Make a plot vs true plot
            plot_pred_vs_true(mode, y_train, y_train_pred, y_tst, y_tst_pred,
                              y_aug, y_aug_pred, suffix)
        
        print(f'{mode} trained successfully...')

    return

###############################################################################
# EXTRA FUNCTIONS
###############################################################################

def train_model(X, y, X_aug, y_aug, mode):

    # Concatenate the training data
    X_trn = pd.concat(X)
    y_trn = pd.concat(y)
    
    # If in data augmentation mode, split the augmented data too
    if mode == 'data_aug':
        X_trn, y_trn, sample_weight = \
            data_aug_split(X_trn, y_trn, X_aug, y_aug)
    else:
        sample_weight = None

    # Create a model with the best hyperparameters
    best_model = Model(mode)
    best_model.load_hps()
    best_model.create_model(X[0].shape[1], 0) # Change 0 to the number of crocus variables

    # Train the model
    history = best_model.fit(X = X_trn, y = y_trn, sample_weight = sample_weight)

    # If the model is a neural network, save the training history
    if best_model.get_model_type() == 'nn' or best_model.get_model_type() == 'lstm':

        # Save the training history
        history_df = pd.DataFrame(history.history)
        history_df.to_csv(paths.outputs / f'train_history_{mode}.csv')
                          
        # Plot the MSE history of the training
        plt.figure()
        plt.plot(history.history['loss'], label='loss')
        plt.legend()
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.savefig(paths.figures / f'train_history_{mode}.png')

    return best_model

###############################################################################

def plot_pred_vs_true(mode, y_train, y_train_pred, y_test, y_test_pred,
                      y_aug=None, y_aug_pred=None, suffix=''):

    # Create scatter plot for training data
    fig = plt.figure(figsize=(12, 7))

    white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
        (0, '#ffffff'),
        (1e-20, '#440053'),
        (0.2, '#404388'),
        (0.4, '#2a788e'),
        (0.6, '#21a784'),
        (0.8, '#78d151'),
        (1, '#fde624'),
    ], N=256)

    ax = fig.add_subplot(1, 2, 1 if mode == 'data_aug' else 1)
    ax.set_aspect('equal', adjustable='box')
    min_val, max_val = np.percentile(np.concatenate([y_train, y_train_pred]), [1, 99])
    density = ax.hist2d(y_train, y_train_pred, bins=range(int(min_val), int(max_val) + 1), cmap=white_viridis)
    fig.colorbar(density[3], ax=ax, label='Number of points per bin')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Train Data')
    plt.plot([min_val, max_val], [min_val, max_val], 'k-', lw=1)

    # Calculate R-squared value and add it to the plot
    r2 = r2_score(y_train, y_train_pred)
    ax.text(0.05, 0.95, f'R-squared = {r2:.2f}', transform=ax.transAxes, fontsize=14,
            verticalalignment='top')

    if mode == 'data_aug':
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.set_aspect('equal', adjustable='box')
        min_val, max_val = np.percentile(np.concatenate([y_aug, y_aug_pred]), [1, 99])
        density = ax2.hist2d(y_aug, y_aug_pred, bins=range(int(min_val), int(max_val) + 1), cmap=white_viridis)
        fig.colorbar(density[3], ax=ax2, label='Number of points per bin')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title('Augmented Data')
        plt.plot([min_val, max_val], [min_val, max_val], 'k-', lw=1)

        # Calculate R-squared value and add it to the plot
        r2 = r2_score(y_aug, y_aug_pred)
        ax2.text(0.05, 0.95, f'R-squared = {r2:.2f}', transform=ax2.transAxes, fontsize=14,
                verticalalignment='top')

    plt.tight_layout()
    name = f'pred_vs_true_{mode}_{suffix}.png' if suffix else f'pred_vs_true_{mode}.png'
    plt.savefig(paths.figures / name)

    # Create scatter plot for test data
    fig_test = plt.figure(figsize=(12, 7))

    ax_test = fig_test.add_subplot(1, 1, 1)
    ax_test.set_aspect('equal', adjustable='box')
    min_val_test, max_val_test = np.percentile(np.concatenate([y_test, y_test_pred]), [1, 99])
    density_test = ax_test.hist2d(y_test, y_test_pred, bins=range(int(min_val_test), int(max_val_test) + 1), cmap=white_viridis)
    fig_test.colorbar(density_test[3], ax=ax_test, label='Number of points per bin')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Test Data')
    plt.plot([min_val_test, max_val_test], [min_val_test, max_val_test], 'k-', lw=1)

    # Calculate R-squared value and add it to the plot
    r2_test = r2_score(y_test, y_test_pred)
    ax_test.text(0.05, 0.95, f'R-squared = {r2_test:.2f}', transform=ax_test.transAxes, fontsize=14,
        verticalalignment='top')

    plt.tight_layout()
    name = f'pred_vs_true_tst_{mode}_{suffix}.png' if suffix else f'pred_vs_true_tst_{mode}.png'
    plt.savefig(paths.figures / name)

    # Save the true and predicted values as csv
    train_df = pd.DataFrame({'TrueValues': y_train, 'PredictedValues': y_train_pred})
    name = f'pred_vs_true_{mode}_{suffix}.csv' if suffix else f'pred_vs_true_{mode}.csv'
    train_df.to_csv(paths.outputs / name, index=False)

    if mode == 'data_aug':
        aug_df = pd.DataFrame({'TrueValues': y_aug, 'PredictedValues': y_aug_pred})
        name = f'pred_vs_true_{mode}_aug_{suffix}.csv' if suffix else f'pred_vs_true_{mode}_aug.csv'
        aug_df.to_csv(paths.outputs / name, index=False)

    # Save the true and predicted values as csv
    test_df = pd.DataFrame({'TrueValues': y_test, 'PredictedValues': y_test_pred})
    name = f'pred_vs_true_tst_{mode}_{suffix}.csv' if suffix else f'pred_vs_true_tst_{mode}.csv'
    test_df.to_csv(paths.outputs / name, index=False)

    return

###############################################################################

def temporal_test_split(X, y, split_idx):

    # Load the split dates
    df_split_dates = pd.read_csv(paths.temp_data / 'split_dates.csv', index_col=[0, 1])

    # Initialize lists to store the training and validation data
    X_trn, y_trn, X_tst, y_tst = [], [], [], []

    for station in cfg.trn_stn:

        # Retrieve the split dates for the current station and split
        trn_val_split_date, val_tst_split_date, tst_trn_split_date = \
            df_split_dates.loc[(station, split_idx)].values
        
        # Filter the trn and tst data conditions for the current station and split
        trn_cond = (X[station].index < val_tst_split_date) | \
                   (X[station].index >= tst_trn_split_date)
        tst_cond = (X[station].index >= val_tst_split_date) & \
                   (X[station].index < tst_trn_split_date)

        # Append the training and test data
        X_trn.append(X[station].loc[trn_cond])
        y_trn.append(y[station].loc[trn_cond])
        X_tst.append(X[station].loc[tst_cond])
        y_tst.append(y[station].loc[tst_cond])          

    return X_trn, X_tst, y_trn, y_tst