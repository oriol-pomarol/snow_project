import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import r2_score
from config import cfg, paths
from modules.auxiliary_functions import mask_measurements_by_year

def simulation_analysis():

    # Make the plots for the predicted vs true dSWE values
    for mode in cfg.modes().keys():

        # Load the train, test and augmented dataframes
        df_trn = pd.read_csv(paths.temp / f'pred_vs_true_{mode}.csv')
        df_tst = pd.read_csv(paths.temp / f'pred_vs_true_tst_{mode}.csv')
        df_aug = None
        if mode == 'data_aug':
            df_aug = pd.read_csv(paths.temp / f'pred_vs_true_{mode}_aug.csv')

        # Make a plot of the predicted vs true values
        plot_pred_vs_true(mode, df_trn, df_tst, df_aug)

    # Load the split dates and convert them to datetime
    df_split_dates = pd.read_csv(paths.temp / 'split_dates.csv', index_col=[0, 1])
    df_split_dates = df_split_dates.apply(pd.to_datetime)

    # Load the measured and simulated snow data for each station
    dict_dfs = {}
    for station_name in cfg.station_names:
        # Load the obs and mod data
        filename = f"df_{station_name}_lag_{cfg.lag}.csv"
        df_obs = pd.read_csv(paths.proc_data / filename, index_col=0)

        # Subset only the SWE columns
        df_obs = df_obs[["obs_swe", "mod_swe"]]

        # Load the ML simulated data
        filename = f"df_{station_name}_pred_swe.csv"
        df_sim = pd.read_csv(paths.temp / filename, index_col=0)

        # Join both dataframes by index
        df_station = df_obs.join(df_sim, how="outer")
        df_station.index = pd.to_datetime(df_station.index)

        # Add the data to the dictionary
        dict_dfs[station_name] = df_station

    # If cfg.station_years is not empty, create a folder with the figures
    if cfg.station_years:
        (paths.figures / 'fwd_sim').mkdir(exist_ok=True)

    # Define the station years to plot
    station_years = ('cdp_2000', 'cdp_2001', 'cdp_2002', 'cdp_2003', 'cdp_2004', 'cdp_2005', 'cdp_2006', 'cdp_2007', 'cdp_2008', 'cdp_2009', 'cdp_2010','rme_1990', 'rme_1991', 'rme_1992', 'rme_1993', 'rme_1994', 'rme_1995', 'rme_1996', 'rme_1997', 'rme_1998', 'rme_1999', 'rme_2000','sod_2007', 'sod_2008', 'sod_2009', 'sod_2010', 'sod_2011', 'sod_2012','oas_all', 'obs_all', 'ojp_all', 'sap_all', 'snb_all', 'swa_all', 'wfj_all', 'all_all')

    for station_year in station_years:

        station_name, year = station_year.split("_")

        # Initialize the figure depending on the number of stations
        if station_name == 'all':
            station_names_plot = cfg.station_names
            fig, axs = plt.subplots(5, 2, figsize=(20, 25))
            axs = axs.flatten()
        
        else:
            station_names_plot = [station_name]
            fig, axs = plt.subplots(1, 1, figsize=(20, 7))
            axs = [axs]

        # Create a dictionary with the dataframes to plot, and a list of labels
        dict_dfs_plot = {st_name: dict_dfs[st_name] for st_name in station_names_plot}
        labels = []

        for station_idx, (station_name, df_station) in enumerate(dict_dfs_plot.items()):
            
            # Get the axis
            ax = axs[station_idx]

            # Get the split dates if temporal split is enabled
            if cfg.temporal_split and (station_name in cfg.trn_stn):
                tst_start_date = df_split_dates.loc[(station_name, 0), 'tst_start_date']
                tst_end_date = df_split_dates.loc[(station_name, 0), 'tst_end_date']
                split_dates = tst_start_date, tst_end_date
            else:
                split_dates = None

            # Mask the measurements and find the number of data points
            df_masked = mask_measurements_by_year(df_station, year, split_dates)
            df_masked_clean = df_masked.dropna()
            num_data_points = df_masked_clean['obs_swe'].count()

            # Plot the observed SWE and calculate the nNSE
            for column_name in df_masked.columns:
                clean_column = df_masked[column_name].dropna()                
                ax.plot(clean_column.index, clean_column,
                        label=column_name)

            # Create the legend
            ax.legend(fontsize='large')
            handles, labels = ax.get_legend_handles_labels()
            handles.append(Line2D([0], [0], marker='None', color='white', label=f'Data points: {num_data_points}'))
            labels.append(f'Data points: {num_data_points}')
            ax.legend(handles=handles, labels=labels, fontsize='large')

            ax.set_xlabel('Date')
            ax.set_ylabel('SWE')
            ax.set_title(f'{station_name.upper()} {year}')
            plt.savefig(paths.figures / 'fwd_sim' / f'{station_name}_{year}.png')
            plt.close()

    return

###############################################################################
# PLOT FUNCTIONS
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