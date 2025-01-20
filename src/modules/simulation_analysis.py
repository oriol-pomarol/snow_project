import pandas as pd
import numpy as np
import calendar
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

    # Start a list to store the test station's dataframes
    test_stations_dfs = []

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

            # Append df to the list of test stations if it is a train station
            test_stations_dfs.append(df_test)
        
        # Calculate the nNSE for the whole station if in station split
        else:
            for metric in metrics:
                metric.calculate_and_append(dict_dfs[station_name], station_name)

            # Append df to the list of test stations if it is a test station
            if (not cfg.temporal_split) and (station_name in cfg.tst_stn):
                test_stations_dfs.append(dict_dfs[station_name])
    
    # Concatenate the DataFrames for the test stations
    df_test = pd.concat(test_stations_dfs)

    # Calculate the metrics for all observations and save the results
    for metric in metrics:
        metric.calculate_and_append(df_test, 'TEST')
        metric.save()

    # If cfg.station_years is not empty, create a folder with the figures
    if cfg.station_years:

        # Create a folder named fwd_sim in the figures directory
        (paths.figures / 'fwd_sim').mkdir(exist_ok=True)

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
            plt.savefig(paths.figures / 'fwd_sim' / f'{station_name}_{year}.png')

    # Plot the error depending on the month of the year for each station
    for station_name in cfg.station_names:
        
        # Get the data for the station and clean it
        df_station = dict_dfs[station_name]
        df_clean = df_station.dropna()

        # Obtain the difference between the observed and simulated SWE
        residuals = {}
        rel_residuals = {}
        for column_name in df_clean.columns:
            if column_name != 'obs_swe':
                residuals[column_name] = abs(df_clean['obs_swe'] - df_clean[column_name])
                rel_residuals[column_name] = df_clean.apply(
                    lambda row: 0 if row['obs_swe'] == 0 else abs(row['obs_swe'] - row[column_name]) / row['obs_swe'], axis=1)
        df_res = pd.DataFrame(residuals, index=df_clean.index)
        df_rel = pd.DataFrame(rel_residuals, index=df_clean.index)

        # Average the the residuals by the month of the year
        df_monthly_res = df_res.groupby(df_res.index.month).mean()
        df_monthly_rel = df_rel.groupby(df_rel.index.month).mean()

        # Map numerical month values to month names and reorder to start from July
        months = [calendar.month_abbr[i] for i in range(7, 13)] + [calendar.month_abbr[i] for i in range(1, 7)]

        # Plot the number of measurements by month
        obs_count = df_clean['obs_swe'].groupby(df_clean.index.month).count()

        # Fill the missing months with zeros and reorder to start from July
        obs_count = obs_count.reindex([7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6], fill_value=0)
        obs_count.index = months

        # Plot the results
        fig, ax = plt.subplots(figsize=(10, 5))
        obs_count.plot(ax=ax, color='black', alpha=0.5)
        plt.xlabel('Month')
        plt.ylabel('Number of measurements')
        ax.set_xticks(range(len(months)))
        ax.set_xticklabels(months, rotation=45)
        plt.savefig(paths.figures / f'{station_name}_monthly_count.png')

        # Reorder the index to start from July
        df_monthly_res = df_monthly_res.reindex([7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]).fillna(0)
        df_monthly_rel = df_monthly_rel.reindex([7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]).fillna(0)

        # Plot the results
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

        # First subplot for df_monthly_res
        for column_name in df_monthly_res.columns:
            ax1.plot(months, df_monthly_res[column_name], label=column_name)

        ax1.legend(fontsize='large')
        ax1.set_ylabel('Error')
        ax1.tick_params(axis='x', rotation=45)

        # Second subplot for df_monthly_rel
        for column_name in df_monthly_rel.columns:
            ax2.plot(months, df_monthly_rel[column_name], label=column_name)

        ax2.set_xlabel('Month')
        ax2.set_ylabel('Relative Error')
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(paths.figures / f'{station_name}_monthly_error.png')

        # Plot the number of measurements by bin
        df_bins = df_res.copy()
        df_bins['obs_swe'] = pd.cut(df_clean['obs_swe'], bins=10)
        df_bins_count = df_bins.groupby('obs_swe').count().fillna(0)

        # Ensure only one column is plotted and rename it to 'Obs'
        df_bins_count = df_bins_count.iloc[:, [0]].rename(columns={df_bins_count.columns[0]: 'Obs'})

        # Plot the data
        fig, ax1 = plt.subplots(figsize=(10, 5))
        df_bins_count.plot(ax=ax1, color='black', alpha=0.5)

        # Add the axis labels
        ax1.set_xlabel('Observed SWE')
        ax1.set_ylabel('Number of measurements')

        # Align the axis labels to the right and ensure all bins are labeled
        ax1.set_xticks(range(len(df_bins_count)))
        ax1.set_xticklabels(df_bins_count.index, rotation=45, ha='right')

        # Hide legend
        ax1.get_legend().remove()

        plt.savefig(paths.figures / f'{station_name}_binned_count.png')

        # Average the residuals by bins of the observed SWE
        df_bins = df_res.copy()
        df_bins['obs_swe'] = pd.cut(df_clean['obs_swe'], bins=10)
        df_bins = df_bins.groupby('obs_swe').mean()

        # Average the relative residuals by bins of the observed SWE
        df_rel_bins = df_rel.copy()
        df_rel_bins['obs_swe'] = pd.cut(df_clean['obs_swe'], bins=10)
        df_rel_bins = df_rel_bins.groupby('obs_swe').mean()

        # Plot the results
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

        # First subplot for df_bins
        for column_name in df_bins.columns:
            ax1.plot(df_bins.index.astype(str), df_bins[column_name], label=column_name)

        ax1.legend(fontsize='large')
        ax1.set_ylabel('Error')

        # Second subplot for df_rel_bins
        for column_name in df_rel_bins.columns:
            ax2.plot(df_rel_bins.index.astype(str), df_rel_bins[column_name], label=column_name)

        ax2.set_xlabel('Observed SWE')
        ax2.set_ylabel('Relative Error')

        # Align the axis labels to the right
        for label in ax1.get_xticklabels():
            label.set_ha('right')
            label.set_rotation(45)

        for label in ax2.get_xticklabels():
            label.set_ha('right')
            label.set_rotation(45)

        plt.tight_layout()
        plt.savefig(paths.figures / f'{station_name}_binned_error.png')

    return

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
    if len(obs) < 2:
        return float('nan')
    return 1 - np.sum((obs - sim) ** 2) / np.sum((obs - np.mean(obs)) ** 2)

###############################################################################

class Metric:
    def __init__(self, name, func, sim_modes):
        self.name = name
        self.func = func
        self.df = pd.DataFrame(columns = sim_modes + ['n'])
        self.sim_modes = sim_modes

    def calculate_and_append(self, data, name):
        self.df.loc[name] = [self.func(data['obs_swe'], data[mode]) for mode in self.sim_modes] + [len(data)]

    def save(self):
        self.df.to_csv(paths.outputs / f'fwd_sim_{self.name}.csv')