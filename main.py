# Import libraries
import time
import os
import pandas as pd
import numpy as np
from modules.data_preprocessing import data_preprocessing
from modules.data_loading import data_loading
from modules.model_training import model_training
from modules.forward_simulation import forward_simulation
# from modules.data_analysis import data_analysis

# Decide what functionality to use
#WIP

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

# Choose what dfs can be used for testing and what only for observations
dfs_test_idx = [1,2,3,5,6,8,9]
dfs_obs_train_idx = [0,4,7]

# Train the models with the different setups
print('Training models...')
# Direct prediction
X = pd.concat([dfs_meteo_agg[j].loc[dfs_obs[j].index] for j in dfs_obs_train_idx])
y = pd.concat([dfs_obs[j] for j in dfs_obs_train_idx])
model_training(X=X, y=y, name='dir_pred')

# Error correction
X = pd.concat([pd.concat([dfs_meteo_agg[j].loc[dfs_obs[j].index],
                          dfs_mod[j].loc[dfs_obs[j].index]], axis=1) for j in dfs_obs_train_idx])
y = pd.concat([dfs_obs[j] for j in dfs_obs_train_idx])
model_training(X=X, y=y, name='err_corr')

# # Data augmentation
# for i in dfs_test_idx:
#     X_obs = pd.concat([dfs_meteo_agg[j].loc[dfs_obs[j].index] for j in dfs_obs_train_idx])
#     X_mod = pd.concat([dfs_meteo_agg[j].loc[dfs_mod[j].index] for j in dfs_test_idx if j!=i])
#     y = pd.concat([pd.concat([dfs_obs[j] for j in dfs_obs_train_idx]),
#                   pd.concat([dfs_mod[j] for j in dfs_test_idx if j!=i])])
#     weight_train_mod = weight_mod * len(X_obs) / len(X_mod)
#     sample_weight = np.concatenate((np.ones(len(X_obs)), np.full(len(X_mod), weight_train_mod)))
#     model_training(X=pd.concat([X_obs,X_mod]), y=y, sample_weight=sample_weight, name=f'data_aug_{i}')
print('Successfully trained models...')

# Test the models by use of the forward simulation
station_names = ['cdp', 'oas', 'obs', 'ojp', 'rme', 'sap', 'snb', 'sod', 'swa', 'wfj']
for i in range(10):
    forward_simulation(dfs_obs[i], dfs_mod[i], dfs_meteo_agg[i], dfs_mod_delta_swe_all[i], 
                       ['nn_dir_pred.h5', 'rf_dir_pred.joblib'], station_names[i])

# Print execution time
end_time = time.time()
execution_time = (end_time - start_time)/60
print('Script finalized.\nExecution time: {:.3g} minutes.'.format(execution_time),
      '\nEnd time: {}'.format(time.ctime(end_time)))