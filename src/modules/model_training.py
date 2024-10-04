import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import r2_score
from config import cfg, paths
from .model_class import Model
from .auxiliary_functions import (
    load_processed_data,
    temporal_data_split,
    data_aug_split
)

def model_training():

    # Load the processed data from all stations
    all_dfs = load_processed_data()
    
    # Store the training and augmentation dataframes and drop NAs
    trn_dfs = [all_dfs[station].dropna() for station in cfg.trn_stn]
    ignore_cols = ["delta_obs_swe", "obs_swe", "res_mod_swe"]
    dropna_cols = [col for col in trn_dfs[0].columns if col not in ignore_cols]
    aug_dfs = [all_dfs[stn].dropna(subset=dropna_cols) for stn in cfg.aug_stn]

    # Filter the biased delta SWE values
    trn_dfs = [df.query('delta_obs_swe != -obs_swe') for df in trn_dfs]
    aug_dfs = [df.query('delta_mod_swe != -mod_swe') for df in aug_dfs]
    
    # Set a random seed for tensorflow
    tf.random.set_seed(10)

    # Define the test data
    if cfg.temporal_split:
        trn_dfs, tst_dfs = temporal_data_split(trn_dfs)

    else:
        tst_dfs = [all_dfs[station] for station in cfg.tst_stn]

    for mode, mode_vars in cfg.modes().items():
    
        # Train the best model for the direct prediction setup
        print(f'Starting {mode} training...')
        
        # Take the corresponding predictor and target variables
        X_obs = [df.filter(regex=mode_vars['predictors']) for df in trn_dfs]
        y_obs = [df[[mode_vars['target']]] for df in trn_dfs]

        # Take the augmented data if in the corresponding mode
        if mode == 'data_aug':
            X_aug = [df.filter(regex='^met_') for df in aug_dfs]
            y_aug = [df[['delta_mod_swe']] for df in aug_dfs]
        else:
            X_aug, y_aug = None, None
            
        # Train the best model and save it
        model = train_model(X_obs, y_obs, X_aug, y_aug, mode = mode)
        model.save_model()

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
        plot_pred_vs_true(mode, y_train, y_train_pred,
                          y_tst, y_tst_pred, y_aug, y_aug_pred)
        
        print(f'{mode} trained successfully...')

    return

###############################################################################
# EXTRA FUNCTIONS
###############################################################################

def train_model(X, y, X_aug, y_aug, mode):

    # Split the data into training and validation sets
    if cfg.temporal_split:
        # Take the last 10% of the data as validation
        X_val = pd.concat([df.iloc[-int(len(df) * 0.1):, :] for df in X])
        y_val = pd.concat([df.iloc[-int(len(df) * 0.1):, :] for df in y])
        X_trn = pd.concat([df.iloc[:-int(len(df) * 0.1), :] for df in X])
        y_trn = pd.concat([df.iloc[:-int(len(df) * 0.1), :] for df in y])

    else:
        # Split the data randomly into training and validation sets
        X_trn, X_val, y_trn, y_val = \
            train_test_split(pd.concat(X), pd.concat(y), test_size=0.2, random_state=10)
    
    # If in data augmentation mode, split the augmented data too
    if mode == 'data_aug':
        X_trn, y_trn, X_val_aug, y_val_aug, sample_weight = \
                data_aug_split(X_trn, y_trn, X_aug, y_aug)
        
    else:
        X_val_aug, y_val_aug, sample_weight = None, None, None

    # Create a model with the best hyperparameters
    best_model = Model(mode)
    best_model.load_hps()
    best_model.create_model(X[0].shape[1], 0) # Change 0 to the number of crocus variables

    # Train the model
    history = best_model.fit(X = X_trn, y = y_trn, X_val = X_val, y_val = y_val,
                             X_aug = X_val_aug, y_aug = y_val_aug, sample_weight = sample_weight)

    # If the model is a neural network, save the training history
    if best_model.get_model_type() == 'nn' or best_model.get_model_type() == 'lstm':

        # Save the training history
        history_df = pd.DataFrame(history.history)
        history_df.to_csv(paths.outputs / f'train_history_{mode}.csv')
                          
        # Plot the MSE history of the training
        plt.figure()
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.legend()
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.savefig(paths.figures / f'train_history_{mode}.png')

    return best_model

###############################################################################

def plot_pred_vs_true(mode, y_train, y_train_pred, y_test, y_test_pred,
                      y_aug=None, y_aug_pred=None):

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
    plt.savefig(paths.figures / f'pred_vs_true_{mode}.png')

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
    plt.savefig(paths.figures / f'pred_vs_true_test_{mode}.png')

    # Save the true and predicted values as csv
    train_df = pd.DataFrame({'TrueValues': y_train, 'PredictedValues': y_train_pred})
    train_df.to_csv(paths.outputs / f'pred_vs_true_{mode}.csv', index=False)

    if mode == 'data_aug':
        aug_df = pd.DataFrame({'TrueValues': y_aug, 'PredictedValues': y_aug_pred})
        aug_df.to_csv(paths.outputs / f'pred_vs_true_{mode}_aug.csv', index=False)

    # Save the true and predicted values as csv
    test_df = pd.DataFrame({'TrueValues': y_test, 'PredictedValues': y_test_pred})
    test_df.to_csv(paths.outputs / f'pred_vs_true_test_{mode}.csv', index=False)

    return