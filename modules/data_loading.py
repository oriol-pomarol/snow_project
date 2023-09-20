import xarray as xr
import os
import pandas as pd

def data_loading(obs_data, meteo_data):
    # List to store the DataFrames
    dfs_meteo = []
    dfs_obs = []

    # Create dictionaries to store variable information for meteo and obs datasets
    meteo_variables = {}
    obs_variables = {}

    # Iterate over the file names
    for name_meteo, name_obs in zip(meteo_data['File name'], obs_data['File name']):

        # Open the NetCDF file using xarray from the URL
        dataset_meteo = xr.open_dataset(os.path.join('data','ESM-SnowMIP_all', f'{name_meteo}.nc'))
        dataset_obs = xr.open_dataset(os.path.join('data','ESM-SnowMIP_all', f'{name_obs}.nc'))

        # Access the variable information for meteo dataset
        for var_name, var in dataset_meteo.variables.items():
            explanation = var.attrs.get('long_name', 'No explanation available')
            units = var.attrs.get('units', '')
            variable_info = f"{var_name}: {explanation} ({units})"
            meteo_variables[var_name] = variable_info

        # Access the variable information for obs dataset
        for var_name, var in dataset_obs.variables.items():
            explanation = var.attrs.get('long_name', 'No explanation available')
            units = var.attrs.get('units', '')
            variable_info = f"{var_name}: {explanation} ({units})"
            obs_variables[var_name] = variable_info

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

        # Remove the first value (as it does not have full meteo input)
        df_obs = df_obs[1:]

        # Drop the hour from the index
        df_obs.index = pd.to_datetime(df_obs.index.date)

        # Append the DataFrames to the corresponding list
        dfs_meteo.append(df_meteo.dropna())
        dfs_obs.append(df_obs.dropna())

    # # Save variable information for the meteo dataset to a file
    # with open(os.path.join('results', 'variables_meteo.txt'), "w") as meteo_file:
    #     meteo_file.write("Variable information for meteo dataset:\n")
    #     for var_name, variable_info in meteo_variables.items():
    #         meteo_file.write(str(variable_info) + "\n")

    # # Save variable information for the obs dataset to a file
    # with open(os.path.join('results', 'variables_obs.txt'), "w") as obs_file:
    #     obs_file.write("Variable information for obs dataset:\n")
    #     for var_name, variable_info in obs_variables.items():
    #         obs_file.write(str(variable_info) + "\n")

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

        # # Create a dictionary to store variable information
        # variable_info = {}

        # Iterate over the variables in the dataset
        for var_name, var in mod_data.variables.items():
            explanation = var.attrs.get('long_name', 'No explanation available')
            units = var.attrs.get('units', '')

            if var_name == 'time':
                start_dates.append(pd.to_datetime(units[12:]))

        #     # Extract the variable number
        #     var_num = ''
        #     for c in var_name[::-1]:
        #         if c.isdigit():
        #             var_num = c + var_num
        #         else:
        #             break

        #     if var_num:
        #         var_base_name = var_name[:-len(var_num)]

        #         # Update the variable_info dictionary
        #         if var_base_name not in variable_info:
        #             variable_info[var_base_name] = {
        #                 'explanations': explanation.replace('1', 'X'),
        #                 'units': units,
        #                 'numbers': []
        #             }
        #         variable_info[var_base_name]['numbers'].append(int(var_num))
        #     else:
        #         variable_info[var_name] = {
        #             'explanations': explanation,
        #             'units': units
        #         }

        # if idx == 0:
        #     with open(os.path.join('results', 'variables_obs.txt'), "w") as mod_file:
        #         mod_file.write("Variable information for mod dataset:\n")
        #         # Get each variable explanation and units
        #         for var_name, info in variable_info.items():
        #             explanation = info['explanations']
        #             units = info['units']
        #             # If there are numerated instances of the same variable, save once specifying the range
        #             if 'numbers' in info:
        #                 var_name += f"{{{min(info['numbers'])}-{max(info['numbers'])}}}"
        #             # Format the information and write to file
        #             variable_info_text = f"{var_name}: {explanation} ({units})"
        #             mod_file.write(variable_info_text + "\n")

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

        # Remove the first value (as it does not have full meteo input)
        df_mod = df_mod[1:]

        # Store the dataframe into the list
        dfs_model.append(df_mod)

    return dfs_obs, dfs_meteo, dfs_model