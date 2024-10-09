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
    aug_dfs = [all_dfs[station].dropna() for station in cfg.aug_stn]

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

        # Define whether to perform data augmentation cross-validation
        aug_cross_val = (mode == 'data_aug') and (not cfg.temporal_split)

        # Generate dataframes to store the predictions
        df_trn = pd.DataFrame(columns=['y_trn', 'y_trn_pred', 'split'])
        df_tst = pd.DataFrame(columns=['y_tst', 'y_tst_pred', 'split'])
        if aug_cross_val:
            df_aug = pd.DataFrame(columns=['y_aug', 'y_aug_pred', 'split'])

        for i in range(len(tst_dfs)) if aug_cross_val else range(1):

            if aug_cross_val:
                # Select all but one station for augmentation
                X_aug = [df.filter(regex='^met_') for j, df in \
                            enumerate(aug_dfs) if j != i]
                y_aug = [df[['delta_mod_swe']] for j, df in \
                            enumerate(aug_dfs) if j != i]
                
                # Select the one station for testing
                X_tst = tst_dfs[i].filter(regex=mode_vars['predictors'])
                y_tst = tst_dfs[i][[mode_vars['target']]]

            else:
                # No data augmentation
                X_aug, y_aug = None, None

                # Select all stations for testing
                X_tst = [df.filter(regex=mode_vars['predictors']) for df in tst_dfs]
                y_tst = [df[[mode_vars['target']]] for df in tst_dfs]
            
            # Train the model
            model = train_model(X_obs, y_obs, X_aug, y_aug, mode = mode)
            model.save_model(suffix=f'aug_split_{i}' if mode == 'data_aug' else '')

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

            # Concatenate the values into the corresponding dataframes
            df_trn = pd.concat([df_trn, pd.DataFrame({'y_trn': y_train,
                                                      'y_trn_pred': y_train_pred,
                                                      'split': i})])
            df_tst = pd.concat([df_tst, pd.DataFrame({'y_tst': y_tst,
                                                      'y_tst_pred': y_tst_pred,
                                                      'split': i})])
            if aug_cross_val:
                df_aug = pd.concat([df_aug, pd.DataFrame({'y_aug': y_aug,
                                                          'y_aug_pred': y_aug_pred,
                                                          'split': i})])
        # Save the dataframes
        df_trn.to_csv(paths.outputs / f'pred_vs_true_{mode}_trn.csv', index=False)
        df_tst.to_csv(paths.outputs / f'pred_vs_true_{mode}_tst.csv', index=False)
        if aug_cross_val:
            df_aug.to_csv(paths.outputs / f'pred_vs_true_{mode}_aug.csv', index=False)

        # Make a plot vs true plot
        plot_pred_vs_true(mode, df_trn, df_tst, df_aug)
        
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

def plot_pred_vs_true(mode, df_trn, df_tst, df_aug=None):

    # Extract the true and predicted values
    y_train = df_trn['y_trn']
    y_train_pred = df_trn['y_trn_pred']
    y_test = df_tst['y_tst']
    y_test_pred = df_tst['y_tst_pred']

    if mode == 'data_aug':
        y_aug = df_aug['y_aug']
        y_aug_pred = df_aug['y_aug_pred']

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
    
    return