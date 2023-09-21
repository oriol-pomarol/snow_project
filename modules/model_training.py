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

def model_training(dfs_obs_delta_swe, dfs_meteo_agg, dfs_mod_delta_swe, lag, dfs_meteo_agg_aug, dfs_mod_delta_swe_aug):

    # Set a random seed for tensorflow
    tf.random.set_seed(10)

    # Choose what dfs can be used for testing and what only for observations
    dfs_obs_train_idx = [0,4,7]

    # Direct prediction
    print('Starting direct prediction training...')

    # Find the common indices for the observations
    common_indices = []
    for df_meteo, df_obs in zip(dfs_meteo_agg, dfs_obs_delta_swe):
        common_indices_i = sorted(set(df_meteo.index).intersection(df_obs.index))
        common_indices.append(common_indices_i)

    # Set the X and y and initialize model selection
    X_obs = [dfs_meteo_agg[j].loc[common_indices[j]] for j in dfs_obs_train_idx]
    y_obs = [dfs_obs_delta_swe[j].loc[common_indices[j]] for j in dfs_obs_train_idx]
    # model_dp = model_selection(X=X_obs, y=y_obs, lag=lag, mode = 'dir_pred')
    print('Direct prediction trained successfully...')

    # Error correction
    print('Starting error correction training...')
    X = [pd.concat([dfs_meteo_agg[j].loc[common_indices[j]],
                    dfs_mod_delta_swe[j].loc[common_indices[j]]], axis=1) \
                        for j in dfs_obs_train_idx]
    model_ec = model_selection(X=X, y=y_obs, lag=lag, mode = 'err_corr')
    print('Error correction trained successfully...')

    # Data augmentation
    print('Starting data augmentation training...')

    # Find the common indices for the augmented data
    common_indices = []
    for df_meteo, df_obs in zip(dfs_meteo_agg_aug, dfs_mod_delta_swe_aug):
        common_indices_i = sorted(set(df_meteo.index).intersection(df_obs.index))
        common_indices.append(common_indices_i)

    # Set the X and y and initialize model selection
    X_aug = [dfs_meteo_agg_aug[j].loc[common_indices[j]] for j in range(len(dfs_meteo_agg_aug))]
    y_aug = [dfs_mod_delta_swe_aug[j].loc[common_indices[j]] for j in range(len(dfs_meteo_agg_aug))]

    model_da = model_selection(X=X_obs, y=y_obs, lag=lag, X_aug=X_aug, y_aug=y_aug, mode = 'data_aug')
    print('Data augmentation trained successfully...')

    # Move any files in the models folder to an old_files folder
    source_folder = os.path.join('results', 'models')
    move_old_files(source_folder)

    # # Save the models
    # for model, mode in zip([model_dp, model_ec, model_da],['dir_pred', 'err_corr', 'data_aug']):
    #     if 'rf' in str(model):
    #         joblib.dump(model.model, os.path.join(source_folder, f'{mode}.joblib'))
    #     elif 'nn' in str(model):
    #         model.model.save(os.path.join(source_folder, f'{mode}.h5'))
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
    learning_rate_vals = [1e-3] #1e-2, 1e-4
    rel_weight_vals = [1] #0.1, 1, 10

    # # Initialize a RF model for each combination of HP
    # for max_depth in max_depth_vals:
    #     for max_samples in max_samples_vals:
    #         if mode == 'data_aug':
    #             for rel_weight in rel_weight_vals:
    #                 model = Model(mode, 'rf', lag)
    #                 model.set_hyperparameters(max_depth=max_depth, max_samples=max_samples,
    #                                           rel_weight=rel_weight)
    #                 models.append(model)
    #         else:
    #             model = Model(mode, 'rf', lag)
    #             model.set_hyperparameters(max_depth=max_depth, max_samples=max_samples)
    #             models.append(model)

    # # Initialize a NN model for each combination of HP
    # for layers in layers_nn_vals:
    #     for learning_rate in learning_rate_vals:
    #         if mode == 'data_aug':
    #             for rel_weight in rel_weight_vals:
    #                 model = Model(mode, 'nn', lag)
    #                 model.set_hyperparameters(layers=layers, learning_rate=learning_rate,
    #                                           rel_weight=rel_weight)
    #                 models.append(model)
    #         else:
    #             model = Model(mode, 'nn', lag)
    #             model.set_hyperparameters(layers=layers, learning_rate=learning_rate)
    #             models.append(model)
            
    # Initialize a LSTM model for each combination of HP
    for layers in layers_lstm_vals:
        for learning_rate in learning_rate_vals:
            if mode == 'data_aug':
                for rel_weight in rel_weight_vals:
                    model = Model(mode, 'lstm', lag)
                    model.set_hyperparameters(layers=layers, learning_rate=learning_rate,
                                              rel_weight=rel_weight)
                    model.create_model(X[0].shape[1])
                    models.append(model)
            else:
                model = Model(mode, 'lstm', lag)
                model.set_hyperparameters(layers=layers, learning_rate=learning_rate)
                model.create_model(X[0].shape[1])
                models.append(model)

    # Perform leave-one-out validation between training stations
    losses = np.zeros((len(models), len(X)))
    hyperparameters = []

    for m, model in enumerate(models):
        hyperparameters.append(str(model))
        print(f'Model {m+1} of {len(models)}.')

        for i in range(len(X)):
            print(f'Train/val split {i+1} of {len(X)}.')
            X_train = pd.concat([X[j] for j in range(len(X)) if j!=i])
            y_train = pd.concat([y[j] for j in range(len(y)) if j!=i])
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=10)
            if mode == 'data_aug':
                len_X_obs_train = len(X_train)
                len_X_aug_train = len(pd.concat(X_aug))
                X_train = pd.concat([X_train, pd.concat(X_aug)])
                y_train = pd.concat([y_train, pd.concat(y_aug)])
                weight_aug = model.hyperparameters.get('rel_weight', 1) * len_X_obs_train / len_X_aug_train
                sample_weight = np.concatenate((np.ones(len_X_obs_train), np.full(len_X_aug_train, weight_aug)))
                
            else:
                sample_weight = None

            model.fit(X=X_train.values, y=y_train.values, X_val=X_val.values,
                      y_val=y_val.values, sample_weight=sample_weight)
            loss = model.test(X=X[i].values, y=y[i].values)
            losses[m,i] = loss

    # Select the best model
    mean_loss = np.mean(losses, axis=1)
    best_model = models[np.argmin(mean_loss)]

    # Save the model hyperparameters and their losses as a csv
    df_losses = pd.DataFrame({'MSE (Split 1)':losses[:,0], 'MSE (Split 2)':losses[:,1],
                              'MSE (Split 3)':losses[:,2], 'MSE (mean)':mean_loss,
                              'HP':hyperparameters})
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
        weight_aug = model.hyperparameters.get('rel_weight', 1) * len_X_obs_train / len_X_aug_train
        sample_weight = np.concatenate((np.ones(len_X_obs_train), np.full(len_X_aug_train, weight_aug)))
    else:
        sample_weight = None

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
                    x = keras.layers.LSTM(units, activation=activation, return_sequences=True)(x)
                else:
                    x = keras.layers.LSTM(units, activation=activation)(x)
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
                value_str = f"{value:.4f}"
                param_name = 'lr'
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