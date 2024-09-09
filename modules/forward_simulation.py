import numpy as np
import pandas as pd
import joblib
from tensorflow import keras
import os
from config import cfg

def forward_simulation():

    # Load the preprocessed data
    dict_dfs = {}
    for station_name in cfg.station_names:
        # Load the data
        df_station = pd.read_csv(
            os.path.join(
                "data",
                "preprocessed",
                f"data_daily_lag_{cfg.lag}",
                f"df_{station_name}_lag_{cfg.lag}.csv",
            ), index_col=0
        )

        # Add the data to the dictionary
        dict_dfs[station_name] = df_station
    
    # Load the models
    list_models = []
    for mode in ['dir_pred', 'err_corr', 'data_aug']:
        model, model_type = load_model(mode)
        list_models.append(Model(mode, model, model_type, cfg.lag))

    # Simulate SWE for each station
    for station_idx, (station_name, df_station) in enumerate(dict_dfs.items()):
        print(f"Simulating station {station_idx+1} of {len(cfg.station_names)}.")

        # Initialize a vector for the predicted SWE
        drop_cols_X = ["obs_swe", "delta_obs_swe", "mod_swe"]
        df_station_X = df_station.drop(drop_cols_X, axis=1).dropna()
        pred_swe_arr = np.zeros((len(list_models), len(df_station_X)))

        for model_idx, model in enumerate(list_models):
            print(f"Mode {model_idx + 1} of {len(list_models)}.")

            for row_idx, row in enumerate(df_station_X.itertuples(index=False,
                                                                  name=None)):
                # Print progress
                if row_idx % (len(df_station_X) // 5) == 0:
                    progress_pct = row_idx * 100 / len(df_station_X)
                    print(f"Progress: {progress_pct:.0f}% completed.")
                
                # Predict the next SWE value
                pred_y = model.predict(row)
                pred_swe = max(pred_swe_arr[model_idx,row_idx-1] + pred_y, 0)
                pred_swe_arr[model_idx, row_idx] = pred_swe
    
        # Save the simulated SWE as a dataframe
        df_swe = pd.DataFrame(pred_swe_arr.T, index=df_station_X.index, 
                              columns=['dir_pred', 'err_corr', 'data_aug'])
        save_path = os.path.join('results', 'simulated_swe')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        df_swe.to_csv(os.path.join(save_path, f'df_{station_name}_sim_swe.csv'))

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

def preprocess_row_lstm(X):

    # Reshape the array by splitting it along the last axis
    transformed_X = X.reshape((X.shape[0] // cfg.lag, cfg.lag))

    # Transpose the subarrays to get the desired structure
    transposed_X = np.transpose(transformed_X)

    return transposed_X

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
        
    def predict(self, tuple_X_row):

        # Preprocess the data for the dir_pred and data_aug modes
        if (self.mode == 'dir_pred') or (self.mode == 'data_aug'):
            X_row = np.array(tuple_X_row[:-1])
            if self.model_type == 'lstm':
                X_row = preprocess_row_lstm(X_row)
            X_row = np.expand_dims(X_row, axis=0)

        # Preprocess the data for the err_corr mode
        elif self.mode == 'err_corr':
            if self.model_type == 'lstm':
                meteo_data = np.array(tuple_X_row[:-1])
                meteo_X = preprocess_row_lstm(meteo_data)
                X_row = [meteo_X, tuple_X_row[-1]]
            else:
                X_row = np.expand_dims(np.array(tuple_X_row), axis=0)

        # Return the predicted delta SWE for this row
        y_pred = self.model.predict(X_row).ravel()
        return y_pred