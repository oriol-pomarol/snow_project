import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import r2_score
from config import cfg, paths
from .auxiliary_functions import find_temporal_split_dates
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
    aug_dfs = [dropna_aug(all_dfs[stn]) for stn in cfg.aug_stn]
    tst_dfs = [all_dfs[station].dropna() for station in cfg.tst_stn]

    # # Filter the biased delta SWE values
    # trn_dfs = [df.query('delta_obs_swe != -obs_swe') for df in trn_dfs]
    # aug_dfs = [df.query('delta_mod_swe != -mod_swe') for df in aug_dfs]
    # tst_dfs = [df.query('delta_obs_swe != -obs_swe') for df in tst_dfs]

    # Find temporal splits if in temporal mode
    if cfg.temporal_split:
        find_temporal_split_dates(trn_dfs)
    
    # Set a random seed for tensorflow
    tf.random.set_seed(10)

    for mode, mode_vars in cfg.modes().items():
        print(f'Starting {mode} training...')
                    
        # Define the number of cross validation splits, with default 1
        n_splits = 1

        # If in temporal mode, use the number of temporal splits
        if cfg.temporal_split:
            n_splits = cfg.n_temporal_splits

        # If in data augmentation mode, use the number of test stations
        elif mode == 'data_aug':
            n_splits = len(cfg.tst_stn)

        # Initialize a df to store the predictions
        df_trn = pd.DataFrame(columns=['y_trn', 'y_trn_pred', 'split'])
        df_tst = pd.DataFrame(columns=['y_tst', 'y_tst_pred', 'split'])
        df_aug = pd.DataFrame(columns=['y_aug', 'y_aug_pred', 'split'])

        for s in range(n_splits):
            print(f'Starting split {s+1} of {n_splits}...')
            
            # Take the corresponding predictor and target variables
            X_obs = [df.filter(regex=mode_vars['predictors']) for df in trn_dfs]
            y_obs = [df[[mode_vars['target']]] for df in trn_dfs]

            # Split the data into training and test sets
            if cfg.temporal_split:
                X_trn, X_tst, y_trn, y_tst = temporal_test_split(X_obs, y_obs, s)
                suffix = f'temp_split_{s}'
            else:
                X_trn, y_trn = X_obs, y_obs
                if mode == 'data_aug':
                    X_tst = [tst_dfs[s].filter(regex=mode_vars['predictors']),]
                    y_tst = [tst_dfs[s][[mode_vars['target']]],]
                    suffix = f'aug_split_{s}'

                else:
                    X_tst = [df.filter(regex=mode_vars['predictors']) for df in tst_dfs]
                    y_tst = [df[[mode_vars['target']]] for df in tst_dfs]
                    suffix = ''

            # Take the augmented data if in the corresponding mode
            X_aug, y_aug = None, None
            if mode == 'data_aug':
                if cfg.temporal_split:
                    X_aug = [df.filter(regex='^met_') for df in aug_dfs]
                    y_aug = [df[['delta_mod_swe']] for df in aug_dfs]
                else:
                    X_aug = [df.filter(regex='^met_') for i, df in enumerate(aug_dfs) if i != s]
                    y_aug = [df[['delta_mod_swe']] for i, df in enumerate(aug_dfs) if i != s]
            
            # Train the model
            model = train_model(X_trn, y_trn, X_aug, y_aug, mode = mode)
            model.save_model(suffix=suffix)

            # Predict the classes for the training and test data
            y_trn_pred = model.predict_classifier(pd.concat(X_trn)).ravel()
            y_tst_pred = model.predict_classifier(pd.concat(X_tst)).ravel()

            # Concatenate the observed classes and convert to 1D numpy array
            y_trn_class = (pd.concat(y_trn).values == 0).astype(int).ravel()
            y_tst_class = (pd.concat(y_tst).values == 0).astype(int).ravel()

            # Make a confusion matrix
            confusion_matrix_trn = tf.math.confusion_matrix(y_trn_class, y_trn_pred).numpy()
            confusion_matrix_tst = tf.math.confusion_matrix(y_tst_class, y_tst_pred).numpy()

            # Make a plot of the confusion matrices
            from sklearn.metrics import ConfusionMatrixDisplay
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            ConfusionMatrixDisplay(confusion_matrix_trn, display_labels=['non-0 values ', 'zero values']).plot(ax=ax[0])
            ax[0].set_title('Training Data')
            ConfusionMatrixDisplay(confusion_matrix_tst, display_labels=['non-0 values ', 'zero values']).plot(ax=ax[1])
            ax[1].set_title('Test Data')
            
            # Save the plot
            plt.tight_layout()
            plt.savefig(paths.figures / f'confusion_matrix_{mode}_{suffix}.png')           

            # Predict the delta SWE for the training and test data
            y_trn_pred = model.predict(pd.concat(X_obs)).ravel()
            y_tst_pred = model.predict(pd.concat(X_tst)).ravel()

            # If in data augmentation, predict delta SWE for the augmented data
            if mode == 'data_aug':
                y_aug_pred = model.predict(pd.concat(X_aug)).ravel()
                y_aug = pd.concat(y_aug).values.ravel()

            # Concatenate the observed values and convert to 1D numpy array
            y_trn = pd.concat(y_obs).values.ravel()
            y_tst = pd.concat(y_tst).values.ravel()

            # Append the predictions to the dataframes
            df_trn = pd.concat([df_trn, pd.DataFrame({'y_trn': y_trn, 'y_trn_pred': y_trn_pred, 'split': s})], ignore_index=True)
            df_tst = pd.concat([df_tst, pd.DataFrame({'y_tst': y_tst, 'y_tst_pred': y_tst_pred, 'split': s})], ignore_index=True)
            if mode == 'data_aug':
                df_aug = pd.concat([df_aug, pd.DataFrame({'y_aug': y_aug, 'y_aug_pred': y_aug_pred, 'split': s})], ignore_index=True)

        # Save the dataframes
        df_trn.to_csv(paths.temp_data / f'pred_vs_true_{mode}.csv', index=False)
        df_tst.to_csv(paths.temp_data / f'pred_vs_true_tst_{mode}.csv', index=False)
        if mode == 'data_aug':
            df_aug.to_csv(paths.temp_data / f'pred_vs_true_{mode}_aug.csv', index=False)
        
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

    # Create a classifier model and train it
    best_model.create_classifier()
    best_model.fit_classifier(X_trn, y_trn)

    # Count the number of meteo variables
    n_met_vars = sum([1 for col in X_trn.columns if col.startswith('met_')])

    # Create the model and fit it to the data
    best_model.create_model(X_trn.shape[1], n_met_vars)

    # Train the model
    history = best_model.fit(X = X_trn, y = y_trn, sample_weight = sample_weight)

    # If the model is a neural network, save the training history
    if best_model.get_model_type() == 'nn' or best_model.get_model_type() == 'lstm':

        # Save the training history
        history_df = pd.DataFrame(history.history)
        history_df.to_csv(paths.temp_data / f'train_history_{mode}.csv')
                          
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

###############################################################################

def temporal_test_split(X, y, split_idx):

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

###############################################################################

def dropna_aug(aug_df):
    ignore_cols = ["delta_obs_swe", "obs_swe", "res_mod_swe"]
    dropna_cols = [col for col in aug_df.columns if col not in ignore_cols]
    clean_df = aug_df.dropna(subset=dropna_cols)
    return clean_df