import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
import joblib

def model_training(dfs_meteo_agg, dfs_mod, dfs_obs_delta_swe):

    # Set a random seed for tensorflow
    tf.random.set_seed(10)

    # Choose what dfs can be used for testing and what only for observations
    # dfs_test_idx = [1,2,3,5,6,8,9]
    dfs_obs_train_idx = [0,4,7]

    # Direct prediction
    X = [dfs_meteo_agg[j].loc[dfs_obs_delta_swe[j].index] for j in dfs_obs_train_idx]
    y = [dfs_obs_delta_swe[j] for j in dfs_obs_train_idx]
    model_dp = model_selection(X=X, y=y)

    # Error correction
    X = [pd.concat([dfs_meteo_agg[j].loc[dfs_obs_delta_swe[j].index],
                    dfs_mod[j].loc[dfs_obs_delta_swe[j].index]], axis=1) \
                        for j in dfs_obs_train_idx]
    y = [dfs_obs_delta_swe[j] for j in dfs_obs_train_idx]
    model_ec = model_selection(X=X, y=y)

    # Data augmentation
    # Set the weight of the modelled values as a whole compared to the observations
    # weight_mod = 0.5
    # for i in dfs_test_idx:
    #     X_obs = pd.concat([dfs_meteo_agg[j].loc[dfs_obs[j].index] for j in dfs_obs_train_idx])
    #     X_mod = pd.concat([dfs_meteo_agg[j].loc[dfs_mod[j].index] for j in dfs_test_idx if j!=i])
    #     y = pd.concat([pd.concat([dfs_obs[j] for j in dfs_obs_train_idx]),
    #                   pd.concat([dfs_mod[j] for j in dfs_test_idx if j!=i])])
    #     weight_train_mod = weight_mod * len(X_obs) / len(X_mod)
    #     sample_weight = np.concatenate((np.ones(len(X_obs)), np.full(len(X_mod), weight_train_mod)))
    #     train_model(X=pd.concat([X_obs,X_mod]), y=y, sample_weight=sample_weight, name=f'data_aug_{i}')
    #     models[model_name] = model

    # Move any files in the models folder to an old_files folder
    source_folder = os.path.join('results', 'models')
    move_old_files(source_folder)

    # Save the models
    for model, mode in zip([model_dp, model_ec],['dir_pred', 'err_corr']):
        if 'rf' in str(model):
            joblib.dump(model.model, os.path.join(source_folder, f'{mode}.joblib'))
        elif 'nn' in str(model):
            model.model.save(os.path.join(source_folder, f'{mode}.h5'))
    return

####################################################################################
# EXTRA FUNCTIONS AND CLASSES
####################################################################################

def model_selection(X, y, sample_weight=None):

    # Initialize the models in a list
    models = []

    # Set the possible values for each hyperparameter
    max_depth_vals = [None, 10, 20]
    max_samples_vals = [None] #, 0.5, 0.8
    layers_vals = [[32], [128]] #, [64,64], [32, 32, 32], [128, 128, 128]
    learning_rate_vals = [1e-2, 1e-4]

    # Initialize a RF model for each combination of HP
    for i, max_depth in enumerate(max_depth_vals):
        for j, max_samples in enumerate(max_samples_vals):
            model = Model('rf')
            model.set_hyperparameters(max_depth=max_depth, max_samples=max_samples)
            model.create_model(X[0].shape[1])
            models.append(model)

    # Initialize a NN model for each combination of HP
    for layers in layers_vals:
        for learning_rate in learning_rate_vals:
            model = Model('nn')
            model.set_hyperparameters(layers=layers, learning_rate=learning_rate)
            model.create_model(X[0].shape[1])
            models.append(model)

    # Perform leave-one-out validation between training stations
    losses = np.zeros((len(models), len(X)))
    for m, model in enumerate(models):
        print(f'Model {m+1} of {len(models)}.')
        for i in range(len(X)):
            print(f'Train/val split {i+1} of {len(X)}.')
            model.fit(X=pd.concat([X[j] for j in range(len(X)) if j!=i]).values, 
                      y=pd.concat([y[j] for j in range(len(X)) if j!=i]).values,
                      sample_weight=sample_weight) # Need to change sample weight to the right subset
            loss = model.test(X=X[i].values, y=y[i].values)
            losses[m,i] = loss

    # Select the best model
    mean_loss = np.mean(losses, axis=1)
    best_model = models[np.argmin(mean_loss)]

    # Train the best model on all the data
    best_model.fit(X=pd.concat(X).values, y=pd.concat(y).values, sample_weight=sample_weight)

    
    # # Plot the MSE history of the training
    # plt.figure()
    # plt.plot(history.history['loss'], label='loss')
    # plt.plot(history.history['val_loss'], label='val_loss')
    # # plt.yscale('log')
    # plt.legend()
    # plt.yscale('log')
    # plt.xlabel('Epoch')
    # plt.ylabel('MSE')
    # plt.savefig(os.path.join('results',f'train_history_{name}.png'))

    return best_model

class Model:
    def __init__(self, model_type):
        if model_type.lower() in ['nn', 'rf']:
            self.model_type = model_type.lower()
            self.model = None
        else:
            raise ValueError("Invalid model type. Choose 'nn' for neural network or 'rf' for random forest.")
        
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
                               loss='mean_squared_error', metrics=['mean_squared_error'])
        elif self.model_type == 'rf':
            self.model = RandomForestRegressor(n_estimators=200, random_state=10, **self.hyperparameters)

    def fit(self, X, y, **kwargs):      
        if self.model_type == 'nn':
            # Define early stopping callback
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
            self.model.fit(X, y, epochs=100, validation_split=0.1, callbacks=[early_stopping], **kwargs)
        elif self.model_type == 'rf':
            self.model.fit(X, y.ravel(), **kwargs)
    
    def test(self, X, y):
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
