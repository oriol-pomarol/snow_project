import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tensorflow import keras
import os
import pickle

def forward_simulation(dfs_obs, dfs_mod, dfs_meteo_agg, dfs_mod_delta_swe_all, 
                       station_years=[], load_data=False):
    # Store the station names for which forward simulation is performed
    station_names = ['cdp', 'oas', 'obs', 'ojp', 'rme', 'sap', 'snb', 'sod', 'swa', 'wfj']
    n_stations = len(station_names)

    # Determine what modes to run
    modes = ['dir_pred', 'err_corr', 'data_aug']

    if not load_data:
        # Create an empty list for the predictions and a df for the MSE
        pred_list = []
        df_mse = pd.DataFrame(columns=modes)
        mod_mse = []

        for i in range(n_stations):
            print(f"Station {i+1} of {n_stations}.")
            pred_swe_arr, mse_swe_list = make_predictions(dfs_obs[i], dfs_meteo_agg[i],
                                                          dfs_mod_delta_swe_all[i], modes)
            
            # Append the modelled data MSE to a list
            pred_obs = dfs_mod[i].loc[dfs_obs[i].index].values
            mse_mod_swe = mean_squared_error(dfs_obs[i].values, pred_obs)
            mod_mse.append(mse_mod_swe)
        
            # Append the predictions and mse to the list/df
            pred_list.append(pred_swe_arr)
            df_mse.loc[i] = mse_swe_list

        # Add the station names as indices, and the modelled data
        df_mse.index = station_names
        df_mse['modelled'] = mod_mse

        # Save the MSE as csv and the predictions with pickle
        df_mse.to_csv(os.path.join('results', 'fwd_sim_mse.csv'))
        with open(os.path.join('results', 'fwd_sim_pred.pkl'), 'wb') as file:
            pickle.dump(pred_list, file)
    else:
        with open(os.path.join('results', 'fwd_sim_pred.pkl'), 'rb') as file:
            pred_list = pickle.load(file)

    # Plot the results      
    for station_year in station_years:
        if station_year == 'all':
            fig, axs = plt.subplots(5, 2, figsize=(15, 8))
            axs = axs.flatten()
            for i in range(len(station_names)):
                ax = axs[i]
                for j, mode in enumerate(modes):
                    ax.plot(dfs_meteo_agg[i].index, pred_list[i][j], label=mode)
                ax.plot(dfs_obs[i].index, dfs_obs[i].values, label='Observed SWE')
                ax.plot(dfs_mod[i].index, dfs_mod[i].values, label='Modelled SWE')
            ax.set_xlabel('Date')
            ax.set_ylabel('SWE')
            ax.legend()
            plt.savefig(os.path.join('results', 'fwd_sim_all.png'))
        else:
            i = station_names.index(station_year[:3])
            year = int(station_year[4:])
            # Plot the results for the year and station given
            fig, ax = plt.subplots(1,1, figsize=(15, 8))
            for j, mode in enumerate(modes):
                mask = mask_measurements_by_year(dfs_meteo_agg[i], year)
                ax.plot(dfs_meteo_agg[i].index[mask], pred_list[i][j][mask], label=mode)
            mask = mask_measurements_by_year(dfs_obs[i], year)
            ax.plot(dfs_obs[i].index[mask], dfs_obs[i].values.ravel()[mask], label='Observed SWE')
            mask = mask_measurements_by_year(dfs_mod[i], year)
            ax.plot(dfs_mod[i].index[mask], dfs_mod[i].values.ravel()[mask], label='Modelled SWE')
            ax.set_xlabel('Date')
            ax.set_ylabel('SWE')
            ax.legend()
            plt.savefig(os.path.join('results', f'fwd_sim_{station_year}.png'))

    return

####################################################################################
# EXTRA FUNCTIONS
####################################################################################

def make_predictions(obs, meteo_agg, mod_delta_swe_all, modes):
    # Initialize a vector for the predicted and observed SWE
    pred_swe_arr = np.zeros((len(modes), len(meteo_agg)))
    mse_swe_list = []

    # Make the forward simulation
    for i, mode in enumerate(modes):
        print(f"Simulating {mode} mode.")
        # Load the trained model
        files_in_folder = os.listdir(os.path.join('results', 'models'))
        model_name = None
        for file in files_in_folder:
            if mode in file:
                model_name = file
                break
        if model_name == None:
            print(f'Error: No model available for {mode}.')
            continue
        
        if '.joblib' in model_name:
            model = joblib.load(os.path.join('results', 'models', model_name))
        if '.h5' in model_name:
            model = keras.models.load_model(os.path.join('results', 'models', model_name))

        # Define the X according to the model
        if (mode == 'dir_pred') or (mode == 'data_aug'):
            fwd_X = meteo_agg
        elif mode == 'err_corr':
            fwd_X = pd.concat([meteo_agg, mod_delta_swe_all], axis=1)
            fwd_X = fwd_X.dropna()

        for j in range(1,len(meteo_agg)):
            if j % (len(meteo_agg) // 5) == 0:
                print(f"Progress: {j * 100 / len(meteo_agg):.0f}% completed.")
            if '.h5' in model_name:
                pred_y = model.predict(fwd_X.values[[j-1]], verbose=0)
            else:
                pred_y = model.predict(fwd_X.values[[j-1]])
            pred_swe_arr[i,j] = max(pred_swe_arr[i,j-1] + pred_y.ravel(), 0)
        
        # Find the MSE and store it in the list
        pred_obs = pred_swe_arr[i][np.isin(meteo_agg.index, obs.index)]
        mse_swe = mean_squared_error(obs.values, pred_obs)
        mse_swe_list.append(mse_swe)

    return pred_swe_arr, mse_swe_list

def mask_measurements_by_year(df, year):
    start_date = pd.to_datetime(f'{year}-07-01')
    end_date = pd.to_datetime(f'{year + 1}-07-01')
    return (df.index >= start_date) & (df.index < end_date)
