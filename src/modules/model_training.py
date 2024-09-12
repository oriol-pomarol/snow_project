import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import r2_score
from config import cfg, paths
from .model_selection import model_selection
from .auxiliary_functions import (
    load_processed_data,
    temporal_data_split
)

def model_training():

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

    # Define the test data
    if cfg.temporal_split:
        trn_dfs, tst_dfs = temporal_data_split(trn_dfs)

    else:
        tst_dfs = [all_dfs[station] for station in cfg.tst_stn]
    
    print('Starting direct prediction training...')
    # Take the meteorological data as input and the delta SWE as output
    X_obs = [df.filter(regex='^met_') for df in trn_dfs]
    y_obs = [df[['delta_obs_swe']] for df in trn_dfs]
    # X_test = [df.filter(regex='^met_') for df in test_dfs]
    # y_test = [df.iloc[:, -2] for df in test_dfs]
    model_dp = model_selection(X=X_obs, y=y_obs, mode = 'dir_pred')
    model_dp = train_model(model_dp, X_obs, y_obs, mode='dir_pred')
    model_dp.save()
    # plot_pred_vs_true(model=model_dp, X_train=X_obs, y_train=y_obs,
    #                   X_test=X_test, y_test=y_test, mode='dir_pred')
    print('Direct prediction trained successfully...')

    # Obtain the best model for the error correction setup
    print('Starting error correction training...')
    X_obs = [df.filter(regex='^(met_|cro_)') for df in trn_dfs]
    y_obs = [df[['res_mod_swe']] for df in trn_dfs]
    # X_test = [df.iloc[:, :-4].join(df.iloc[:, -1]) for df in test_dfs]
    # y_test = [df.iloc[:, -2] for df in test_dfs]
    model_ec = model_selection(X=X_obs, y=y_obs, mode = 'err_corr')
    model_ec = train_model(model_ec, X_obs, y_obs, mode = 'err_corr')
    model_ec.save()
    # plot_pred_vs_true(model=model_ec, X_train=X_obs, y_train=y_obs,
    #                   X_test=X_test, y_test=y_test, mode='err_corr')
    print('Error correction trained successfully...')

    # Obtain the best model for the data augmentation setup
    print('Starting data augmentation training...')
    X_obs = [df.filter(regex='^met_') for df in trn_dfs]
    y_obs = [df[['delta_obs_swe']] for df in trn_dfs]
    X_aug = [df.filter(regex='^met_') for df in aug_dfs]
    y_aug = [df[['delta_mod_swe']] for df in aug_dfs]
    # X_test = [df.iloc[:, :-4] for df in test_dfs]
    # y_test = [df.iloc[:, -2] for df in test_dfs]
    model_da = model_selection(X=X_obs, y=y_obs, X_aug=X_aug,
                               y_aug=y_aug, mode = 'data_aug')
    model_da = train_model(model_da, X_obs, y_obs, X_aug, y_aug, 'data_aug')
    model_da.save()
    # plot_pred_vs_true(model=model_da, X_train=X_obs, y_train=y_obs,
    #                   X_test=X_test, y_test=y_test, mode='data_aug',
    #                   X_aug=X_aug, y_aug=y_aug)
    print('Data augmentation trained successfully...')

    return

###############################################################################
# EXTRA FUNCTIONS AND CLASSES
###############################################################################

def train_model(best_model, X, y, X_aug, y_aug, mode):

    # Train the best model on all the data
    X_train, X_val, y_train, y_val = train_test_split(pd.concat(X), pd.concat(y),
                                                      test_size=0.2, random_state=10)
    
    if mode == 'data_aug':
        X_train_aug, y_train_aug = pd.concat(X_aug), pd.concat(y_aug)
        # Define the lengths of the training observed and augmented data 
        len_X_obs_train = len(X_train)
        len_X_aug_train = len(X_train_aug)

        # Concatenate the observed and augmented data and convert to arrays
        X_train_arr = pd.concat([X_train, X_train_aug]).values
        y_train_arr = pd.concat([y_train, y_train_aug]).values

        # Calculate the relative weight of the obs/mod data
        weight_aug = best_model.hyperparameters.get('rel_weight', 1) * len_X_obs_train / len_X_aug_train
        sample_weight = np.concatenate((np.ones(len_X_obs_train), 
                                        np.full(len_X_aug_train, weight_aug)))
        
    else:
        # Convert the dataframes to arrays
        X_train_arr = X_train.values
        y_train_arr = y_train.values
        sample_weight = None

    # Create and fit the best model
    best_model.create_model(X[0].shape[1])
    history = best_model.fit(X=X_train_arr, y=y_train_arr, X_val=X_val.values,
                            y_val=y_val.values, sample_weight=sample_weight)

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

def plot_pred_vs_true(model, X_train, y_train, X_test, y_test, mode,
                      X_aug=None, y_aug=None, y_train_mod=None, y_test_mod=None):
    
    y_train_arr = pd.concat(y_train).values
    y_test_arr = pd.concat(y_test).values

    if mode == 'mod_swe':
        y_train_pred = pd.concat(y_train_mod).values
        y_test_pred = pd.concat(y_test_mod).values

    else:
        # Predict values for training and test data
        X_train_arr = pd.concat(X_train).values
        X_test_arr = pd.concat(X_test).values
        y_train_pred = model.predict(X_train_arr).ravel()
        y_test_pred = model.predict(X_test_arr).ravel()

        if mode == 'data_aug':
            X_aug_arr = pd.concat(X_aug).values
            y_aug_arr = pd.concat(y_aug).values
            y_aug_pred = model.predict(X_aug_arr).ravel()

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
    min_val, max_val = np.percentile(np.concatenate([y_train_arr, y_train_pred]), [1, 99])
    density = ax.hist2d(y_train_arr, y_train_pred, bins=range(int(min_val), int(max_val) + 1), cmap=white_viridis)
    fig.colorbar(density[3], ax=ax, label='Number of points per bin')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Train Data')
    plt.plot([min_val, max_val], [min_val, max_val], 'k-', lw=1)

    # Calculate R-squared value and add it to the plot
    r2 = r2_score(y_train_arr, y_train_pred)
    ax.text(0.05, 0.95, f'R-squared = {r2:.2f}', transform=ax.transAxes, fontsize=14,
            verticalalignment='top')

    if mode == 'data_aug':
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.set_aspect('equal', adjustable='box')
        min_val, max_val = np.percentile(np.concatenate([y_aug_arr, y_aug_pred]), [1, 99])
        density = ax2.hist2d(y_aug_arr, y_aug_pred, bins=range(int(min_val), int(max_val) + 1), cmap=white_viridis)
        fig.colorbar(density[3], ax=ax2, label='Number of points per bin')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title('Augmented Data')
        plt.plot([min_val, max_val], [min_val, max_val], 'k-', lw=1)

        # Calculate R-squared value and add it to the plot
        r2 = r2_score(y_aug_arr, y_aug_pred)
        ax2.text(0.05, 0.95, f'R-squared = {r2:.2f}', transform=ax2.transAxes, fontsize=14,
                verticalalignment='top')

    plt.tight_layout()
    plt.savefig(paths.figures / f'pred_vs_true_{mode}.png')

    # Create scatter plot for test data
    fig_test = plt.figure(figsize=(12, 7))

    ax_test = fig_test.add_subplot(1, 1, 1)
    ax_test.set_aspect('equal', adjustable='box')
    min_val_test, max_val_test = np.percentile(np.concatenate([y_test_arr, y_test_pred]), [1, 99])
    density_test = ax_test.hist2d(y_test_arr, y_test_pred, bins=range(int(min_val_test), int(max_val_test) + 1), cmap=white_viridis)
    fig_test.colorbar(density_test[3], ax=ax_test, label='Number of points per bin')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Test Data')
    plt.plot([min_val_test, max_val_test], [min_val_test, max_val_test], 'k-', lw=1)

    # Calculate R-squared value and add it to the plot
    r2_test = r2_score(y_test_arr, y_test_pred)
    ax_test.text(0.05, 0.95, f'R-squared = {r2_test:.2f}', transform=ax_test.transAxes, fontsize=14,
        verticalalignment='top')

    plt.tight_layout()
    plt.savefig(paths.figures / f'pred_vs_true_test_{mode}.png')

    # Save the true and predicted values as csv
    train_df = pd.DataFrame({'TrueValues': y_train_arr, 'PredictedValues': y_train_pred})
    train_df.to_csv(paths.outputs / f'pred_vs_true_{mode}.csv', index=False)

    if mode == 'data_aug':
        aug_df = pd.DataFrame({'TrueValues': y_aug_arr, 'PredictedValues': y_aug_pred})
        aug_df.to_csv(paths.outputs / f'pred_vs_true_{mode}_aug.csv', index=False)

    # Save the true and predicted values as csv
    test_df = pd.DataFrame({'TrueValues': y_test_arr, 'PredictedValues': y_test_pred})
    test_df.to_csv(paths.outputs / f'pred_vs_true_test_{mode}.csv', index=False)

    return