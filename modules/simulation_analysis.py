import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import mean_squared_error
from config import cfg

def simulation_analysis(station_years=[]):

    if cfg.temporal_split:
        # Load the train/test into a dictionary
        path = os.path.join("results", "split_dates.csv")
        df_split_dates = pd.read_csv(path, index_col=0)
        dict_split_dates = {index: row.tolist() for index, row
                            in df_split_dates.iterrows()}


    # Load the station data
    dict_dfs = {}
    for station_name in cfg.station_names:
        # Load the obs and mod data
        dir_path = os.path.join("data", "preprocessed", f"data_daily_lag_{cfg.lag}")
        filename = f"df_{station_name}_lag_{cfg.lag}.csv"
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

    # Create an empty dataframe for the nNSE
    sim_modes = ['mod_swe', 'dir_pred', 'err_corr', 'data_aug']
    df_nnse = pd.DataFrame(columns=sim_modes)

    # Find the nNSE and store them in the dataframe for each station
    for station_name in cfg.station_names:

        # Clean the data
        df_station_clean = dict_dfs[station_name].dropna()

        if cfg.temporal_split and station_name in dict_split_dates:
            # Split the data into training and testing sets
            split_date = dict_split_dates[station_name]
            df_train = mask_measurements_by_year(df_station_clean, 'train', split_date)
            df_test = mask_measurements_by_year(df_station_clean, 'test', split_date)
        
            # Calculate the nNSE for the training set and append it to the df
            obs_swe_train = df_train['obs_swe']
            nnse_train = [calculate_nNSE(obs_swe_train, df_train[mode]) for mode in sim_modes]
            df_nnse.loc[station_name + '_train'] = nnse_train
            
            # Calculate the nNSE for the testing set and append it to the df
            obs_swe_test = df_test['obs_swe']
            nnse_test = [calculate_nNSE(obs_swe_test, df_test[mode]) for mode in sim_modes]
            df_nnse.loc[station_name + '_test'] = nnse_test
        
        else:
            # Calculate the nNSE of the whole station and append it to the df
            obs_swe_train = df_station_clean['obs_swe']
            nnse_train = [calculate_nNSE(obs_swe_train, df_station_clean[mode]) for mode in sim_modes]
            df_nnse.loc[station_name] = nnse_train

    # Save the nNSE as csv
    df_nnse.to_csv(os.path.join('results', 'fwd_sim_nnse.csv'))

    # Plot the results
    print('Plotting the results...')  
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

            # Mask the measurements and find the number of data points
            if cfg.temporal_split:
                split_date = dict_split_dates.get(station_name, None)
            else:
                split_date = None
            df_masked = mask_measurements_by_year(df_station, year, split_date)
            df_masked_clean = df_masked.dropna()
            num_data_points = df_masked_clean['obs_swe'].count()

            # Plot the observed SWE and calculate the nNSE
            for column_name in df_masked.columns:
                nNSE = calculate_nNSE(df_masked_clean["obs_swe"] ,
                                      df_masked_clean[column_name])
                clean_column = df_masked[column_name].dropna()                
                ax.plot(clean_column.index, clean_column,
                        label=f'{column_name} (nNSE: {nNSE:.2f})')

            # Create the  legend
            ax.legend(fontsize='large')
            handles, labels = ax.get_legend_handles_labels()
            handles.append(Line2D([0], [0], marker='None', color='white', label=f'Data points: {num_data_points}'))
            labels.append(f'Data points: {num_data_points}')
            ax.legend(handles=handles, labels=labels, fontsize='large')

            ax.set_xlabel('Date')
            ax.set_ylabel('SWE')
            ax.set_title(f'{station_name.upper()} {year}')
            plt.savefig(os.path.join('results', f'fwd_sim_{station_name}_{year}.png'))

###############################################################################
# EXTRA FUNCTIONS
###############################################################################

def mask_measurements_by_year(df, year, split_dates=None):

    if year == 'all':
        return df
    
    elif year == 'train':
        start_date, end_date = split_dates
        mask = (df.index < start_date) | (df.index > end_date)

    elif year == 'test':
        start_date, end_date = split_dates
        mask = (df.index >= start_date) & (df.index <= end_date)

    elif year.isdigit(): 
        year = int(year)
        start_date = pd.to_datetime(f'{year}-07-01')
        end_date = pd.to_datetime(f'{year + 1}-07-01')
        mask = (df.index >= start_date) & (df.index < end_date)
        
    else:
        raise ValueError(f'Invalid input year: {year}')
    
    return df[mask]
    
def calculate_nNSE(observed, predicted):
    mse = mean_squared_error(observed, predicted)
    nNSE = 1 / (2 - (1 - mse / np.var(observed)))
    return nNSE