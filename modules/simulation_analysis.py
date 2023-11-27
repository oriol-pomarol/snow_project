import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import mean_squared_error

def simulation_analysis(station_years=[]):
    # Define what lag value to use
    lag = 14

    # List of station names
    station_names = [
        "cdp",
        "oas",
        "obs",
        "ojp",
        "rme",
        "sap",
        "snb",
        "sod",
        "swa",
        "wfj",
    ]

    import pandas as pd

    # Load the train/test into a dictionary
    path = os.path.join("results", "train_test_split.csv")
    df_train_test_split = pd.read_csv(path, index_col=0, parse_dates=True)
    train_test_split_dict = df_train_test_split['train_test_split_dates'].to_dict()

    # Load the station data
    dict_dfs = {}
    for station_name in station_names:
        # Load the obs and mod data
        dir_path = os.path.join("data", "preprocessed", f"data_daily_lag_{lag}")
        filename = f"df_{station_name}_lag_{lag}.csv"
        df_obs = pd.read_csv(os.path.join(dir_path, filename), index_col=0)

        # Subset only the SWE columns
        df_obs = df_obs[["obs_swe", "mod_swe"]]

        # Load the ML simulated data
        dir_path = os.path.join("results", "simulated_swe")
        filename = f"df_{station_name}_sim_swe.csv"
        df_sim = pd.read_csv(os.path.join(dir_path, filename), index_col=0)

        # Join both dataframes by index
        df_station = df_obs.join(df_sim, how="outer")
        df_station.index = pd.to_datetime(df_station.index)

        # Add the data to the dictionary
        dict_dfs[station_name] = df_station

    # # Create an empty df for the MSE and nNSE
    # sim_modes = ['mod_swe', 'dir_pred', 'err_corr', 'data_aug']
    # df_mse = pd.DataFrame(columns=sim_modes)
    # df_nnse = pd.DataFrame(columns=sim_modes)

    # # Find the MSE and nNSE and store them in the lists
    # for station_name in station_names:
    #     df_station_clean = dict_dfs[station_name].dropna()
    #     obs_swe = df_station_clean['obs_swe']
    #     mse_station = [mean_squared_error(obs_swe, df_station_clean[mode])
    #                    for mode in sim_modes]
    #     nnse_station = [1 / (2 - (1 - mse / np.var(obs_swe)))
    #                     for mse in mse_station]

    #     # Append the predictions, MSE and nNSE to the df
    #     df_mse.loc[station_name] = mse_station
    #     df_nnse.loc[station_name] = nnse_station

    #     # Save the MSE and nNSE as csv and the predictions with pickle
    #     df_mse.to_csv(os.path.join('results', 'fwd_sim_mse.csv'))
    #     df_nnse.to_csv(os.path.join('results', 'fwd_sim_nnse.csv'))

    # Plot the results
    print('Plotting the results...')  
    for station_year in station_years:

        station_name, year = station_year.split("_")

        if station_name == 'all':
            station_names_plot = station_names
            fig, axs = plt.subplots(5, 2, figsize=(20, 25))
            axs = axs.flatten()
        
        else:
            station_names_plot = [station_name]
            fig, axs = plt.subplots(1, 1, figsize=(20, 7))
            axs = [axs]

        dict_dfs_plot = {st_name: dict_dfs[st_name] for st_name in station_names_plot}
        labels = []

        for station_idx, (station_name, df_station) in enumerate(dict_dfs_plot.items()):
            ax = axs[station_idx]
            train_test_split = train_test_split_dict.get(station_name, None)
            df_masked = mask_measurements_by_year(df_station, year, train_test_split)
            num_data_points = df_masked['obs_swe'].count()  # Count the number of data points
            for column_name in df_masked.columns:
                clean_column = df_masked[column_name].dropna()
                
                # Calculate nNSE and append it to the legend
                nNSE = calculate_nNSE(df_masked["obs_swe"] , clean_column)                
                ax.plot(clean_column.index, clean_column, label=f'{column_name} (nNSE: {nNSE:.2f})')

            # Create the original legend
            ax.legend(fontsize='large')

            # Get the handles and labels of the original legend
            handles, labels = ax.get_legend_handles_labels()

            # Add the number of data points as a new legend item
            handles.append(Line2D([0], [0], marker='None', color='white', label=f'Data points: {num_data_points}'))
            labels.append(f'Data points: {num_data_points}')

            # Create the new legend
            ax.legend(handles=handles, labels=labels, fontsize='large')

            ax.set_xlabel('Date')
            ax.set_ylabel('SWE')
            ax.set_title(f'{station_name.upper()} {year}')
            plt.savefig(os.path.join('results', f'fwd_sim_{station_name}_{year}.png'))

###############################################################################
# EXTRA FUNCTIONS
###############################################################################

def mask_measurements_by_year(df, year, train_test_split_date=None):
    if year == 'all':
        return df
    elif year == 'train':
        return df[df.index <= train_test_split_date]
    elif year == 'test':
        return df[df.index > train_test_split_date]
    elif year.isdigit(): 
        year = int(year)
        start_date = pd.to_datetime(f'{year}-07-01')
        end_date = pd.to_datetime(f'{year + 1}-07-01')
        return df[(df.index >= start_date) & (df.index < end_date)]
    
def calculate_nNSE(predicted, observed):
    """Calculate the nNSE between two series"""

    # Join the series into a dataframe to remove NaNs
    df = predicted.to_frame().join(observed.to_frame(), how='inner',
                                   lsuffix='_pred', rsuffix='_obs')
    df.columns = ['predicted', 'observed']
    df = df.dropna()

    # Retrieve observed and predicted arrays
    observed_array = df['observed'].to_numpy()
    predicted_array = df['predicted'].to_numpy()

    # Calculate the nNSE
    nNSE = 1 / (2 - (1 - mean_squared_error(observed_array, predicted_array) \
                     / np.var(observed_array)))
    return nNSE