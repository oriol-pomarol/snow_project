import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
import joblib

def model_training():

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


    # Set a random seed for tensorflow and limit the warnings
    tf.random.set_seed(10)
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Define what stations will be used for training, augmenting and testing
    trng_dfs = [dict_dfs["cdp"], dict_dfs["rme"], dict_dfs["sod"]]
    augm_dfs = [dict_dfs["oas"], dict_dfs["obs"], dict_dfs["ojp"],
                dict_dfs["sap"], dict_dfs["snb"], dict_dfs["swa"]]

    # Filter the biased delta SWE values and drop NaNs
    trng_dfs = [df.loc[df['delta_obs_swe'] != -1 * df['obs_swe'], :].dropna()
                for df in trng_dfs]
    augm_dfs = [df.loc[df['delta_mod_swe'] != -1 * df['mod_swe'], :].dropna()
                for df in augm_dfs]

    # Obtain the best model for the direct prediction setup
    print('Starting direct prediction training...')
    X_obs = [df.iloc[:int(len(df)*0.8), :-4] for df in trng_dfs]
    y_obs = [df.iloc[:int(len(df)*0.8), -2] for df in trng_dfs]
    model_dp = model_selection(X=X_obs, y=y_obs, lag=lag, mode = 'dir_pred')
    print('Direct prediction trained successfully...')

    # Obtain the best model for the error correction setup
    print('Starting error correction training...')
    X_obs = [df.iloc[:int(len(df)*0.8), :-4].join(df.iloc[:, -1]) for df in trng_dfs]
    y_obs = [df.iloc[:int(len(df)*0.8), -2] for df in trng_dfs]
    model_ec = model_selection(X=X_obs, y=y_obs, lag=lag, mode = 'err_corr')
    print('Error correction trained successfully...')

    # Obtain the best model for the data augmentation setup
    print('Starting data augmentation training...')
    X_obs = [df.iloc[:int(len(df)*0.8), :-4] for df in trng_dfs]
    y_obs = [df.iloc[:int(len(df)*0.8), -2] for df in trng_dfs]
    X_aug = [df.iloc[:int(len(df)*0.8), :-4] for df in augm_dfs]
    y_aug = [df.iloc[:int(len(df)*0.8), -2] for df in augm_dfs]

    model_da = model_selection(X=X_obs, y=y_obs, lag=lag, X_aug=X_aug,
                               y_aug=y_aug, mode = 'data_aug')
    print('Data augmentation trained successfully...')

    # Move any files in the models folder to an old_files folder
    source_folder = os.path.join('results', 'models')
    move_old_files(source_folder)

    # Save the models
    for model, mode in zip([model_dp, model_ec, model_da],
                           ['dir_pred', 'err_corr', 'data_aug']):
        if 'rf' in str(model):
            joblib.dump(model.model, os.path.join(source_folder, f'{mode}.joblib'))
        elif ('nn' in str(model)) or ('lstm' in str(model)):
            model.model.save(os.path.join(source_folder, f'{mode}.h5'))
    return

####################################################################################
# EXTRA FUNCTIONS AND CLASSES
####################################################################################

def model_selection(X, y, lag, X_aug=[], y_aug=[], mode=''):
    # Initialize the models in a list
    models = []

    # Set the possible values for each hyperparameter
    max_depth_vals = [None, 10, 20]
    max_samples_vals = [None, 0.5, 0.8]
    layers_nn_vals = [[2048], [128, 128, 128]]
    layers_lstm_vals = [[512], [128, 64]]
    learning_rate_vals = [1e-3, 1e-5]
    l2_reg_vals = [0, 1e-2, 1e-4]
    rel_weight_vals = [0.5, 1, 2]

    # Initialize a RF model for each combination of HP
    for max_depth in max_depth_vals:
        for max_samples in max_samples_vals:
            model = Model(mode, 'rf', lag)
            model.set_hyperparameters(max_depth=max_depth,
                                      max_samples=max_samples)
            models.append(model)

    # Initialize a NN model for each combination of HP
    for layers in layers_nn_vals:
        for learning_rate in learning_rate_vals:
            for l2_reg in l2_reg_vals:
                model = Model(mode, 'nn', lag)
                model.set_hyperparameters(layers=layers,
                                          learning_rate=learning_rate,
                                          l2_reg=l2_reg)
                models.append(model)

    # Initialize a LSTM model for each combination of HP
    if lag > 0:
        for layers in layers_lstm_vals:
            for learning_rate in learning_rate_vals:
                for l2_reg in l2_reg_vals:
                    model = Model(mode, 'lstm', lag)
                    model.set_hyperparameters(layers=layers,
                                              learning_rate=learning_rate,
                                              l2_reg=l2_reg)
                    models.append(model)

    # Perform a single validation with a temporal train/val split
    losses = np.zeros(len(models))
    model_names = []

    for m, model in enumerate(models):
        model_names.append(str(model))
        print(f'Model {m+1} of {len(models)}.')

        X_train = pd.concat([df.iloc[:int(len(df)*0.8)] for df in X])
        y_train = pd.concat([df.iloc[:int(len(df)*0.8)] for df in y])
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                            test_size=0.1, random_state=10)
        if mode == 'data_aug':
            len_X_obs_train = len(X_train)
            len_X_aug_train = len(pd.concat(X_aug))
            X_train = pd.concat([X_train, pd.concat(X_aug)])
            y_train = pd.concat([y_train, pd.concat(y_aug)])
            weight_aug = model.hyperparameters.get('rel_weight', 1) * len_X_obs_train / len_X_aug_train
            sample_weight = np.concatenate((np.ones(len_X_obs_train), 
                                            np.full(len_X_aug_train, weight_aug)))
            
        else:
            sample_weight = None

        model.create_model(X_train.shape[1])
        model.fit(X=X_train.values, y=y_train.values, X_val=X_val.values,
                    y_val=y_val.values, sample_weight=sample_weight)
        X_test = pd.concat([df.iloc[int(len(df)*0.8):] for df in X])
        y_test = pd.concat([df.iloc[int(len(df)*0.8):] for df in y])
        loss = model.test(X=X_test.values, y=y_test.values)
        losses[m] = loss

    # Select the best model
    best_model = models[np.argmin(losses)]

    # If in data augmentation, tune the relative weight of the obs/mod data
    if mode == 'data_aug':
        losses_rw = np.zeros(len(rel_weight_vals))
        for w, rel_weight in enumerate(rel_weight_vals):
            model_names.append(str(best_model) + f'_rw_{rel_weight}')
            if rel_weight == 1:
                loss = np.min(losses)
            else:
                print(f'Relative weight {m+1} of {len(models)}.')
                X_train = pd.concat([df.iloc[:int(len(df)*0.8)] for df in X])
                y_train = pd.concat([df.iloc[:int(len(df)*0.8)] for df in y])
                X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
                                                                    test_size=0.1, random_state=10)
                len_X_obs_train = len(X_train)
                len_X_aug_train = len(pd.concat(X_aug))
                X_train = pd.concat([X_train, pd.concat(X_aug)])
                y_train = pd.concat([y_train, pd.concat(y_aug)])
                weight_aug = rel_weight * len_X_obs_train / len_X_aug_train
                sample_weight = np.concatenate((np.ones(len_X_obs_train),
                                                np.full(len_X_aug_train, weight_aug)))
                best_model.create_model(X_train.shape[1])
                model.fit(X=X_train.values, y=y_train.values, X_val=X_val.values,
                            y_val=y_val.values, sample_weight=sample_weight)
                X_test = pd.concat([df.iloc[int(len(df)*0.8):] for df in X])
                y_test = pd.concat([df.iloc[int(len(df)*0.8):] for df in y])
                model.test(X=X_test.values, y=y_test.values)
            losses_rw[w] = loss

        # Select the best model
        best_rw = rel_weight_vals[np.argmin(losses_rw)]
        losses = np.append(losses, losses_rw)

    # Save the model hyperparameters and their losses as a csv
    df_losses = pd.DataFrame({'MSE (mean)':losses,'HP':model_names})
    df_losses.set_index('HP', inplace=True)
    df_losses.to_csv(os.path.join('results', f'model_losses_{mode}.csv'))

    # Train the best model on all the data
    X_train, X_val, y_train, y_val = train_test_split(pd.concat(X), pd.concat(y),
                                                      test_size=0.2, random_state=10)
    if mode == 'data_aug':
        len_X_obs_train = len(X_train)
        len_X_aug_train = len(pd.concat(X_aug))
        X_train = pd.concat([X_train, pd.concat(X_aug)])
        y_train = pd.concat([y_train, pd.concat(y_aug)])
        weight_aug = best_rw * len_X_obs_train / len_X_aug_train
        sample_weight = np.concatenate((np.ones(len_X_obs_train), np.full(len_X_aug_train, weight_aug)))
    else:
        sample_weight = None

    best_model.create_model(X[0].shape[1])
    history = best_model.fit(X=X_train.values, y=y_train.values, X_val=X_val.values,
                             y_val=y_val.values, sample_weight=sample_weight)

    if best_model.get_model_type() == 'nn' or best_model.get_model_type() == 'lstm':
        # Save the training history
        history_df = pd.DataFrame(history.history)
        history_df.to_csv(os.path.join('results', f'train_history_{mode}.csv'))
                          
        # Plot the MSE history of the training
        plt.figure()
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.legend()
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.savefig(os.path.join('results',f'train_history_{mode}.png'))

    return best_model

####################################################################################

class Model:
    def __init__(self, mode, model_type, lag):
        valid_model_type = model_type.lower() in ['nn', 'rf','lstm'] 
        valid_mode = mode.lower() in ['dir_pred', 'err_corr', 'data_aug']
        if valid_model_type and valid_mode:
            self.model_type = model_type.lower()
            self.mode = mode.lower()
            self.lag = lag
            self.model = None
        else:
            raise ValueError("Invalid model setup or model type.")
        
        self.hyperparameters = {}

    def set_hyperparameters(self, **kwargs):
        self.hyperparameters = kwargs
        self.model = None  # Clear any existing model

    def create_model(self, input_shape):
        self.model = None  # Clear any existing model
        if self.model_type == 'nn':
            self.model = keras.Sequential()
            self.model.add(keras.layers.Input(shape=input_shape))
            for units in self.hyperparameters.get('layers', [128]):
                activation = self.hyperparameters.get('activation', 'relu')
                self.model.add(keras.layers.Dense(units, activation=activation))
            self.model.add(keras.layers.Dense(1, activation='linear'))
            self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.hyperparameters.get('learning_rate', 0.001)),
                               loss='mean_squared_error', metrics=['mean_squared_error'], weighted_metrics=[])
        elif self.model_type == 'lstm':
            sequential_input = keras.layers.Input(shape=(self.lag, (input_shape-1*(self.mode=='err_corr')) // self.lag))
            activation = self.hyperparameters.get('activation', 'relu')
            depth = len(self.hyperparameters.get('layers'))
            x = sequential_input
            for layer, units in enumerate(self.hyperparameters.get('layers', [128])):
                if (depth > 1) and (layer + 1 < depth):
                    x = keras.layers.LSTM(units, return_sequences=True)(x)
                else:
                    x = keras.layers.LSTM(units)(x)
            if self.mode == 'err_corr':
                extra_var_input = keras.layers.Input(shape=(1,), name='extra_var_input')
                combined_input = keras.layers.Concatenate()([x, extra_var_input])
                x = keras.layers.Dense(units=128, activation=activation)(combined_input)
            output_layer = keras.layers.Dense(1, activation='linear')(x)
            if self.mode == 'err_corr':
                self.model = keras.models.Model(inputs=[sequential_input, extra_var_input], outputs=output_layer)
            else:
                self.model = keras.models.Model(inputs=sequential_input, outputs=output_layer)
            self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.hyperparameters.get('learning_rate', 0.001)),
                               loss='mean_squared_error', metrics=['mean_squared_error'], weighted_metrics=[])
        elif self.model_type == 'rf':
            self.model = RandomForestRegressor(n_estimators=200, random_state=10,
                                               max_depth=self.hyperparameters.get('max_depth', None),
                                               max_samples=self.hyperparameters.get('max_samples', None))

    def fit(self, X, y, X_val, y_val, **kwargs):
        if self.model_type == 'lstm':
            if self.mode == 'err_corr':
                X_mod = X[:,-1]
                X = X[:,:-1]
                X_val_mod = X_val[:,-1]
                X_val = X_val[:,:-1]
            X = preprocess_data_lstm(X, self.lag)
            X_val = preprocess_data_lstm(X_val, self.lag)      
        if self.model_type in ['nn', 'lstm']:
            # Define early stopping callback
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
            if self.model_type == 'lstm' and self.mode == 'err_corr':
                history = self.model.fit([X,X_mod], y, epochs=100, validation_data=([X_val,X_val_mod], y_val),
                                         callbacks=[early_stopping], **kwargs)
            else:
                history = self.model.fit(X, y, epochs=100, validation_data=(X_val, y_val), callbacks=[early_stopping], **kwargs)
            return history
        elif self.model_type == 'rf':
            self.model.fit(X, y.ravel(), **kwargs)
            return None
    
    def test(self, X, y):
        if self.model_type == 'lstm':
            if self.mode == 'err_corr':
                X_mod = X[:,-1]
                X = X[:,:-1]
            X = preprocess_data_lstm(X, self.lag)
        if self.model_type == 'lstm' and self.mode == 'err_corr':
            y_pred = self.model.predict([X,X_mod])
        else:
            y_pred = self.model.predict(X)
        mse = mean_squared_error(y, y_pred)
        return mse

    def get_model_type(self):
        return self.model_type

    def __str__(self):
        model_name = self.model_type
        for key, value in self.hyperparameters.items():
            param_name = key[:2]  # Take the first two characters of the hyperparameter name
            if key == 'layers':
                value_str = "_".join([f"{unit:03d}" for unit in value])
                param_name = 'ly'
            elif key == 'learning_rate':
                value_str = f"{value:.0e}".replace("-", "_")
                param_name = 'lr'
            elif key == 'l2_reg':
                value_str = f"{value:.0e}".replace("-", "_")
                param_name = 'rg'
            else:
                value_str = str(value)
            model_name += f"_{param_name}{value_str}"
        
        return model_name

####################################################################################

def move_old_files(source_folder):
    target_folder = os.path.join(source_folder, "old_files")
    
    # Create 'old_files' directory if it doesn't exist
    if not os.path.exists(target_folder):
        os.mkdir(target_folder)
    
    # List all files in the source folder
    files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    
    if files:
        print(f"Moving {len(files)} files to 'old_files' folder...")
        for file in files:
            source_path = os.path.join(source_folder, file)
            target_path = os.path.join(target_folder, file)
            os.rename(source_path, target_path)
        print("Files moved successfully.")
    else:
        print("No files to move.")

####################################################################################

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