# Import libraries
import time
import os
import pandas as pd
import numpy as np
from modules.data_preprocessing import data_preprocessing
from modules.data_loading import data_loading
from modules.model_training import model_training
from modules.forward_simulation import forward_simulation

# Set the weight of the modelled values as a whole compared to the observations
weight_mod = 0.5

# Record starting run time
start_time = time.time()

# Read the station data
stations = pd.read_csv(os.path.join('data', 'Menard_Essery_2019.tab'), 
                       delimiter='\t', skiprows=35)

# Save the in-situ meteo and obs data as separate dataframes
met_data = stations[10:20].reset_index()
obs_data = stations[20:30].reset_index()

# Save the longitudes and latitudes of each location
locations = np.column_stack((obs_data['Latitude'].to_numpy(),
                             obs_data['Longitude'].to_numpy()))

# Load data
print('Loading the data...')
dfs_obs, dfs_met, dfs_mod = data_loading(obs_data, met_data)
print('Successfully loaded the data...')

# Preprocess data
print('Pre-processing the data...')
dfs_obs_delta_swe, dfs_meteo_agg, dfs_mod_delta_swe_all, dfs_mod_delta_swe_filt = \
    data_preprocessing(dfs_obs, dfs_met, dfs_mod, locations)
print('Successfully pre-processed the data...')

# Train the models with the different setups
print('Training models...')
model_training(dfs_obs, dfs_meteo_agg, dfs_mod)
print('Successfully trained models...')

# Test the models by use of the forward simulation
print('Performing forward simulation...')
model_names = ['nn_dir_pred', 'rf_dir_pred']
forward_simulation(dfs_obs, dfs_mod, dfs_meteo_agg, dfs_mod_delta_swe_all, model_names, station_year='cdp_2005')
print('Successfully performed forward simulation...')

# Print execution time
end_time = time.time()
execution_time = (end_time - start_time)/60
print('Script finalized.\nExecution time: {:.3g} minutes.'.format(execution_time),
      '\nEnd time: {}'.format(time.ctime(end_time)))