# Import libraries
import time
import os
import pandas as pd
import numpy as np
from modules.data_preprocessing import data_preprocessing
from modules.data_load_preprocess import data_loading_and_preprocessing
from modules.model_training import model_training
from modules.forward_simulation import forward_simulation

# Record starting run time
start_time = time.time()

# Load and preprocess data
dfs_data = data_loading_and_preprocessing()
print('Successfully loaded and preprocessed the data...')

# # Preprocess data
# print('Pre-processing the data...')
# lag, dfs_obs_delta_swe, dfs_meteo_agg, dfs_mod_delta_swe, dfs_mod_delta_swe_filt = \
#     data_preprocessing(dfs_obs, dfs_met, dfs_mod, locations)
# print('Successfully pre-processed the data...')

# # Train the models with the different setups
# print('Training models...')
# model_training(dfs_obs_delta_swe, dfs_meteo_agg, dfs_mod_delta_swe, lag,
#                [dfs_meteo_agg[i] for i in [1,2,3,5,6,8]], 
#                [dfs_mod_delta_swe_filt[i] for i in [1,2,3,5,6,8]])
# print('Successfully trained models...')

# # Test the models by use of the forward simulation
# print('Performing forward simulation...')
# forward_simulation(dfs_obs, dfs_mod, dfs_meteo_agg, dfs_mod_delta_swe,
#                    lag, station_years=['cdp_2002','rme_2002', 'wfj_2002'])
# print('Successfully performed forward simulation...')

# # Print execution time
# end_time = time.time()
# execution_time = (end_time - start_time)/60
# print('Script finalized.\nExecution time: {:.3g} minutes.'.format(execution_time),
#       '\nEnd time: {}'.format(time.ctime(end_time)))
