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

    modes = ['dir_pred', 'err_corr', 'data_aug']

    print('Plotting the results...')
    # Plot the results      
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

    # # Create an empty list for the predictions and a df for the MSE and nNSE
    # df_mse = pd.DataFrame(columns=modes)
    # df_nnse = pd.DataFrame(columns=modes)

# # Find the MSE and nNSE and store them in the lists
# pred_obs = pred_swe_arr[i][np.isin(meteo_agg.index, obs.index)]
# obs_av = obs[np.isin(obs.index, meteo_agg.index)].values
# mse_swe = mean_squared_error(obs_av, pred_obs)
# mse_swe_list.append(mse_swe)
# nnse_swe_list.append(1 / (2 - (1 - mse_swe / np.var(obs_av))))

# # Append the modelled data MSE and nNSE to a list
# pred_obs = dfs_mod[i].loc[dfs_obs[i].index].values
# mse_mod_swe = mean_squared_error(dfs_obs[i].values, pred_obs)
# mod_mse.append(mse_mod_swe)
# mod_nnse.append(1 / (2 - (1 - mse_mod_swe / np.var(dfs_obs[i].values))))

# # Append the predictions, MSE and nNSE to the list/df
# pred_list.append(pred_swe_arr)
# df_mse.loc[i] = mse_swe_list
# df_nnse.loc[i] = nnse_swe_list

# # Add the modelled data
# df_mse['modelled'] = mod_mse
# df_nnse['modelled'] = mod_nnse

# # Save the MSE and nNSE as csv and the predictions with pickle
# df_mse.to_csv(os.path.join('results', 'fwd_sim_mse.csv'))
# df_nnse.to_csv(os.path.join('results', 'fwd_sim_nnse.csv'))
# with open(os.path.join('results', 'fwd_sim_pred.pkl'), 'wb') as file:
#     pickle.dump(pred_list, file)
# else:
# with open(os.path.join('results', 'fwd_sim_pred.pkl'), 'rb') as file:
#     pred_list = pickle.load(file)



def mask_measurements_by_year(df, year):
    if year == 'all':
        return df
    year = int(year)
    start_date = pd.to_datetime(f'{year}-07-01')
    end_date = pd.to_datetime(f'{year + 1}-07-01')
    return df[(df.index >= start_date) & (df.index < end_date)]