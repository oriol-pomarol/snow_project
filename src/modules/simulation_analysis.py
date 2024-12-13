import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from config import cfg, paths

def simulation_analysis():

    # Create a list with the simulation modes and add Crocus simulations
    sim_modes = ['mod_swe'] + list(cfg.modes().keys())

    # Define the metrics
    metrics = [
        Metric('rmse', root_mean_squared_error, sim_modes),
        Metric('mae', mean_absolute_error, sim_modes),
        Metric('mbe', mean_bias_error, sim_modes),
        Metric('nse', nash_sutcliffe_efficiency, sim_modes),
    ]

    if cfg.temporal_split:
        # Load the split dates
        df_split_dates = pd.read_csv(paths.temp / 'split_dates.csv', index_col=[0, 1])

        # Convert all columns to a datetime format
        df_split_dates = df_split_dates.apply(pd.to_datetime)

    # Load the true and simulated snow data for each station
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

        # Clean the data
        df_station = df_station.dropna()

        # Add the data to the dictionary
        dict_dfs[station_name] = df_station

    # Find the nNSE and store them in the dataframe for each station
    for station_name in cfg.station_names:

        # Calculate the nNSE for the training set if in temporal split
        if cfg.temporal_split and (station_name in cfg.trn_stn):

            # Find the start and end dates for the testing set
            tst_start_date = df_split_dates.loc[(station_name, 0), 'tst_start_date']
            tst_end_date = df_split_dates.loc[(station_name, 0), 'tst_end_date']
            split_dates = tst_start_date, tst_end_date

            # Mask the measurements by testing sets
            df_test = mask_measurements_by_year(dict_dfs[station_name], 'test', split_dates)
            
            # Calculate and append the metrics
            for metric in metrics:
                metric.calculate_and_append(df_test, station_name)
        
        # Calculate the nNSE for the whole station if in station split
        else:
            for metric in metrics:
                metric.calculate_and_append(dict_dfs[station_name], station_name)
   
    # Calculate the metrics for all observations and save the results
    df_all = pd.concat(dict_dfs.values(), axis=0)
    for metric in metrics:
        metric.calculate_and_append(df_all, 'TOTAL')
        metric.save()

    # Plot the results
    print('Plotting the results...')  
    for station_year in cfg.station_years:

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
            if cfg.temporal_split:
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
                nse = nash_sutcliffe_efficiency(df_masked_clean["obs_swe"],
                                                df_masked_clean[column_name])
                clean_column = df_masked[column_name].dropna()                
                ax.plot(clean_column.index, clean_column,
                        label=f'{column_name} (NSE: {nse:.2f})')

            # Create the legend
            ax.legend(fontsize='large')
            handles, labels = ax.get_legend_handles_labels()
            handles.append(Line2D([0], [0], marker='None', color='white', label=f'Data points: {num_data_points}'))
            labels.append(f'Data points: {num_data_points}')
            ax.legend(handles=handles, labels=labels, fontsize='large')

            ax.set_xlabel('Date')
            ax.set_ylabel('SWE')
            ax.set_title(f'{station_name.upper()} {year}')
            plt.savefig(paths.figures / f'fwd_sim_{station_name}_{year}.png')

###############################################################################
# EXTRA FUNCTIONS
###############################################################################

def mask_measurements_by_year(df, year, split_dates=None):

    # If the dataframe is empty or the year is 'all', return the dataframe
    if (len(df) == 0) or (year == 'all'):
        return df

    # If the year is 'train', 'test', or a specific year, mask the data
    elif year == 'train':
        mask = (df.index < split_dates[0]) | (df.index >= split_dates[1])

    elif year == 'test':
        mask = (df.index >= split_dates[0]) & (df.index < split_dates[1])

    elif year.isdigit():
        year = int(year)
        start_date = pd.to_datetime(f'{year}-07-01')
        end_date = pd.to_datetime(f'{year + 1}-07-01')
        mask = (df.index >= start_date) & (df.index < end_date)
        
    else:
        raise ValueError(f'Invalid input year: {year}')
    
    return df[mask]

###############################################################################

def root_mean_squared_error(obs, sim):
    if len(obs) < 1:
        return float('nan')
    return np.sqrt(np.mean((obs - sim) ** 2))

def mean_absolute_error(obs, sim):
    if len(obs) < 1:
        return float('nan')
    return np.mean(np.abs(obs - sim))

def mean_bias_error(obs, sim):
    if len(obs) < 1:
        return float('nan')
    return np.mean(obs - sim)

def nash_sutcliffe_efficiency(obs, sim):
    if len(obs) < 1:
        return float('nan')
    return 1 - np.sum((obs - sim) ** 2) / np.sum((obs - np.mean(obs)) ** 2)

###############################################################################

class Metric:
    def __init__(self, name, func, sim_modes):
        self.name = name
        self.func = func
        self.df = pd.DataFrame(columns=sim_modes)
        self.sim_modes = sim_modes

    def calculate_and_append(self, data, name):
        self.df.loc[name] = [self.func(data['obs_swe'], data[mode]) for mode in self.sim_modes]

    def save(self):
        self.df.to_csv(paths.outputs / f'fwd_sim_{self.name}.csv')