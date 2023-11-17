import numpy as np
import pandas as pd
import joblib
from tensorflow import keras
import os

def forward_simulation():
    
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

    # Load the preprocessed data
    dict_dfs = {}
    for station_name in station_names:
        # Load the data
        df_station = pd.read_csv(
            os.path.join(
                "data",
                "preprocessed",
                f"data_daily_lag_{lag}",
                f"df_{station_name}_lag_{lag}.csv",
            ), index_col=0
        )

        # Add the data to the dictionary
        dict_dfs[station_name] = df_station
    
    # Load the models
    list_models = []
    for mode in ['dir_pred', 'err_corr', 'data_aug']:
        model, model_type = load_model(mode)
        list_models.append(Model(mode, model, model_type, lag))

    # Simulate SWE for each station
    for station_idx, station_name in enumerate(station_names):
        print(f"Simulating station {station_idx+1} of {len(station_names)}.")

        # Initialize a vector for the predicted SWE
        drop_cols_X = ["obs_swe", "delta_obs_swe"]
        df_station_wo_obs = df_station.drop(drop_cols_X, axis=1).dropna()
        df_station_X = df_station_wo_obs.drop("mod_swe", axis=1)
        pred_swe_arr = np.zeros((len(list_models), len(df_station_X)))

        for model_idx, model in enumerate(list_models):
            print(f"Simulating {mode} mode.")

            for row_idx, row in enumerate(df_station_X.iterrows()):
                # Print progress
                if row_idx % (len(df_station_X) // 5) == 0:
                    progress_pct = row_idx * 100 / len(df_station_X)
                    print(f"Progress: {progress_pct:.0f}% completed.")
                
                # Predict the next SWE value
                pred_y = model.predict(row, mode=mode)
                pred_swe = max(pred_swe_arr[model_idx,row_idx-1] + pred_y, 0)
                pred_swe_arr[model_idx, row_idx] = pred_swe
    
        # Save the simulated SWE as a transposed dataframe
        df_swe = pd.DataFrame(pred_swe_arr.T, index=df_station_wo_obs.index, 
                              columns=['dir_pred', 'err_corr', 'data_aug'])
        # Append the mod and obs SWE to the df
        df_swe['mod_pred'] = df_station_wo_obs['mod_pred']
        df_swe.to_csv(os.path.join('results', f'df_{station_name}_sim_swe.csv'))

    return

####################################################################################
# EXTRA FUNCTIONS
####################################################################################

def load_model(mode):
    # Find the model filename
    files_in_folder = os.listdir(os.path.join('results', 'models'))
    model_filename = None
    for file in files_in_folder:
        if mode in file:
            model_filename = file
            break
    if model_filename == None:
        print(f'Error: No model available for {mode}.')
        return None, None

    # Load the model
    if '.joblib' in model_filename:
        model_type = 'rf'
        model = joblib.load(os.path.join('results', 'models', model_filename))
    if '.h5' in model_filename:
        model_type = 'nn'
        model = keras.models.load_model(os.path.join('results', 'models', model_filename))
        # Check if the model contains an LSTM layer
        for layer in model.layers:
            if isinstance(layer, keras.layers.LSTM):
                model_type = 'lstm'
                break
    return model, model_type

def preprocess_data_lstm(X, lag):
    # Get the shape of the input array
    shape = X.shape

    # Calculate the number of subarrays along the last axis
    num_subarrays = shape[-1] // lag

    # Reshape the array by splitting it along the last axis
    new_shape = shape[:-1] + (num_subarrays, lag)
    transformed_X = X.reshape(new_shape)

    # Transpose the subarrays to get the desired structure
    transformed_X = np.transpose(transformed_X, axes=(0, 2, 1))

    return transformed_X

class Model:
    def __init__(self, mode, model, model_type, lag):
        valid_model_type = model_type.lower() in ['nn', 'rf','lstm'] 
        valid_mode_type = mode.lower() in ['dir_pred', 'err_corr', 'data_aug']
        if valid_model_type and valid_mode_type:
            self.mode = mode.lower()
            self.model = model
            self.model_type = model_type.lower()
            self.lag = lag
        else:
            raise ValueError("Invalid model type.")
        
    def get_mode(self):
        return self.mode
        
    def predict(self, df_X_row, mode):

        # Preprocess the data for the dir_pred and err_corr modes
        if (mode == 'dir_pred') or (mode == 'err_corr'):
            X_row = df_X_row.drop('mod_swe').values
            if self.model_type == 'lstm':
                X_row = preprocess_data_lstm(X_row, self.lag)

        # Preprocess the data for the data_aug mode
        elif mode == 'data_aug':
            if self.model_type == 'lstm':
                meteo_data = df_X_row.drop('mod_swe').values
                meteo_X = preprocess_data_lstm(meteo_data, self.lag)
                X_row = [meteo_X, df_X_row['mod_swe'].values]
            else:
                X_row = df_X_row.values

        # Return the predicted delta SWE for this row
        y_pred = self.model.predict(X_row).ravel()
        return y_pred