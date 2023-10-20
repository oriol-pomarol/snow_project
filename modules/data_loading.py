import xarray as xr
import os
import pandas as pd

def data_loading(obs_data, meteo_data):
    # List to store the DataFrames
    dfs_meteo = []
    dfs_obs = []

    # Iterate over the file names
    for name_meteo, name_obs in zip(meteo_data['File name'], obs_data['File name']):

        # Open the NetCDF file using xarray from the URL
        dataset_meteo = xr.open_dataset(os.path.join('data','ESM-SnowMIP_all', f'{name_meteo}.nc'))
        dataset_obs = xr.open_dataset(os.path.join('data','ESM-SnowMIP_all', f'{name_obs}.nc'))

        # Convert the dataset to a pandas DataFrame
        df_meteo = dataset_meteo.to_dataframe()
        df_obs = dataset_obs.to_dataframe()

        # Take the available snow water equivalent measurements at each station
        if 'snw_auto' in df_obs.columns:
            df_obs = df_obs[['snw_auto']]
        else:
            df_obs = df_obs[['snw_man']]

        # Take only the first obs sample for each unique time value
        df_obs = df_obs.groupby('time').first()

        # Take only the measurements at 12:00 for the Reynolds station
        if name_obs == 'obs_insitu_rme_1988_2008':
            df_obs = df_obs[df_obs.index.strftime('%H:%M') == '12:00']

        # Drop the hour from the index
        df_obs.index = pd.to_datetime(df_obs.index.date)

        # Append the DataFrames to the corresponding list
        dfs_meteo.append(df_meteo.dropna())
        dfs_obs.append(df_obs.dropna())

    # Initialize the list of dataframes and initial times
    dfs_model = []
    start_dates = []

    # Specify the path to the parent directory containing the folders
    parent_directory = os.path.join('data', 'simus_CROCUS')

    # Get the list of subdirectories (folders) in the parent directory
    subdirectories = sorted([f.path for f in os.scandir(parent_directory)])

    # Iterate over the files
    for idx, subdirectory in enumerate(subdirectories):
        # Determine the file path
        file_path = os.path.join(subdirectory, os.listdir(subdirectory)[0])

        # Read the .nc file into a dataset
        mod_data = xr.open_dataset(file_path, decode_times=False)

        # Find the starting time of each simulation
        time_var = mod_data.variables.get('time')
        start_date = pd.to_datetime(time_var.attrs.get('units')[12:])
        start_dates.append(start_date)

        # Select the total SWE only
        mod_swe_data = mod_data['WSN_T_ISBA']

        # Convert to a DataFrame, store and rename the total SWE
        df_mod = mod_swe_data.to_dataframe()
        df_mod.rename(columns={'WSN_T_ISBA': 'mod_swe'}, inplace=True)

        # Take only the first sample for each unique time value
        df_mod = df_mod.groupby('time').first()

        # Convert the 'time' index to a datetime index starting at the corresponding date
        start_datetime = pd.to_datetime(start_dates[idx])
        num_hours = len(df_mod)
        datetime_index = pd.date_range(start=start_datetime, periods=num_hours, freq='H')
        df_mod.index = datetime_index

        # Take only the measurements made at 12:00
        df_mod = df_mod[df_mod.index.strftime('%H:%M') == '12:00']
        df_mod.index = pd.to_datetime(df_mod.index.date)

        # Store the dataframe into the list
        dfs_model.append(df_mod)

    return dfs_obs, dfs_meteo, dfs_model