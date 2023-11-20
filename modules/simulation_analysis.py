import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
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

    # Create an empty df for the MSE and nNSE
    sim_modes = ['mod_swe', 'dir_pred', 'err_corr', 'data_aug']
    df_mse = pd.DataFrame(columns=sim_modes)
    df_nnse = pd.DataFrame(columns=sim_modes)

    # Find the MSE and nNSE and store them in the lists
    for station_name in station_names:
        df_station_clean = dict_dfs[station_name].dropna()
        obs_swe = df_station_clean['obs_swe']
        mse_station = [mean_squared_error(obs_swe, df_station_clean[mode])
                       for mode in sim_modes]
        nnse_station = [1 / (2 - (1 - mse / np.var(obs_swe)))
                        for mse in mse_station]

        # Append the predictions, MSE and nNSE to the df
        df_mse.loc[station_name] = mse_station
        df_nnse.loc[station_name] = nnse_station

        # Save the MSE and nNSE as csv and the predictions with pickle
        df_mse.to_csv(os.path.join('results', 'fwd_sim_mse.csv'))
        df_nnse.to_csv(os.path.join('results', 'fwd_sim_nnse.csv'))

    # Plot the results
    print('Plotting the results...')  
    for station_year in station_years:
        station_name, year = station_year.split("_")
        if station_name == 'all':
            fig, axs = plt.subplots(5, 2, figsize=(15, 8))
            axs = axs.flatten()
            for station_idx, (station_name, df_station) in enumerate(dict_dfs.items()):
                ax = axs[station_idx]
                df_masked = mask_measurements_by_year(df_station, year)
                for column_name in df_masked.columns:
                    clean_column = df_masked[column_name].dropna()
                    ax.plot(clean_column.index, clean_column, '.', label=column_name)
            plt.legend()
            plt.savefig(os.path.join('results', 'fwd_sim_all.png'))

        else:
            df_station = dict_dfs[station_name]
            df_masked = mask_measurements_by_year(df_station, year)
            fig, ax = plt.subplots(1,1, figsize=(15, 8))
            for column_name in df_masked.columns:
                clean_column = df_masked[column_name].dropna()
                ax.plot(clean_column.index, clean_column, label=column_name)
            ax.set_xlabel('Date')
            ax.set_ylabel('SWE')
            ax.set_title(f'{station_name.upper()} {year}')
            ax.legend()
            plt.savefig(os.path.join('results', f'fwd_sim_{station_year}.png'))

###############################################################################
# EXTRA FUNCTIONS
###############################################################################

def mask_measurements_by_year(df, year):
    if year == 'all':
        return df
    year = int(year)
    start_date = pd.to_datetime(f'{year}-07-01')
    end_date = pd.to_datetime(f'{year + 1}-07-01')
    return df[(df.index >= start_date) & (df.index < end_date)]