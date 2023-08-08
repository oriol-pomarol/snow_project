import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import os
import joblib

def model_training(dfs_meteo_agg, dfs_mod, dfs_obs_delta_swe):

    # Choose what dfs can be used for testing and what only for observations
    # dfs_test_idx = [1,2,3,5,6,8,9]
    dfs_obs_train_idx = [0,4,7]

    # Start a dictionary with models and their names
    models = {}

    # Direct prediction
    X = [dfs_meteo_agg[j].loc[dfs_obs_delta_swe[j].index] for j in dfs_obs_train_idx]
    y = [dfs_obs_delta_swe[j] for j in dfs_obs_train_idx]
    model, model_name = model_selection(X=X, y=y)
    models[model_name] = model

    # Error correction
    X = [pd.concat([dfs_meteo_agg[j].loc[dfs_obs_delta_swe[j].index],
                    dfs_mod[j].loc[dfs_obs_delta_swe[j].index]], axis=1) \
                        for j in dfs_obs_train_idx]
    y = [dfs_obs_delta_swe[j] for j in dfs_obs_train_idx]
    model, model_name = model_selection(X=X, y=y)
    models[model_name] = model

    # Data augmentation
    # Set the weight of the modelled values as a whole compared to the observations
    weight_mod = 0.5
    # for i in dfs_test_idx:
    #     X_obs = pd.concat([dfs_meteo_agg[j].loc[dfs_obs[j].index] for j in dfs_obs_train_idx])
    #     X_mod = pd.concat([dfs_meteo_agg[j].loc[dfs_mod[j].index] for j in dfs_test_idx if j!=i])
    #     y = pd.concat([pd.concat([dfs_obs[j] for j in dfs_obs_train_idx]),
    #                   pd.concat([dfs_mod[j] for j in dfs_test_idx if j!=i])])
    #     weight_train_mod = weight_mod * len(X_obs) / len(X_mod)
    #     sample_weight = np.concatenate((np.ones(len(X_obs)), np.full(len(X_mod), weight_train_mod)))
    #     train_model(X=pd.concat([X_obs,X_mod]), y=y, sample_weight=sample_weight, name=f'data_aug_{i}')
    #     models[model_name] = model

    # Save the models
    for model_name, model in models.items():
        if 'rf' in model_name:
            joblib.dump(model, os.path.join('results', 'models', f'{model_name}.joblib'))
        elif 'nn' in model_name:
            model.save(os.path.join('results', 'models', f'{model_name}.h5'))

####################################################################################
# EXTRA FUNCTIONS AND CLASSES
####################################################################################

def model_selection(X, y, sample_weight=None):

    # Initialize the models in a list
    models = []

    # Set the possible values for each hyperparameter
    max_depth_vals = [None, 10, 20]
    max_samples_vals = [None, 0.5, 0.8]
    layers_vals = [[32], [128], [64,64], [32, 32, 32], [128, 128, 128]]
    learning_rate_vals = [1e-2, 1e-4]

    # Initialize a RF model for each combination of HP
    for max_depth in max_depth_vals:
        for max_samples in max_samples_vals:
            model = Model('rf')
            model.set_hyperparameters(max_depth=max_depth, max_samples=max_samples)
            models.append(model)

    # Initialize a NN model for each combination of HP
    for layers in layers_vals:
        for learning_rate in learning_rate_vals:
            model = Model('nn')
            model.set_hyperparameters(layers=layers, learning_rate=learning_rate)
            models.append(model)

    # Perform leave-one-out validation between training stations
    for i in range(len(X)):
        for model in models:
            model.fit(pd.concat([X[j] for j in range(len(X)) if j!=i]), 
                      pd.concat([y[j] for j in range(len(X)) if j!=i]),
                      sample_weight=sample_weight)

    
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

    return #best_model, best_model_name

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
            self.model = RandomForestRegressor(**self.hyperparameters)

    def fit(self, X, y, **kwargs):
        if not self.model:
            raise ValueError("Please create the model first.")
        
        if self.model_type == 'nn':
            self.model.fit(X, y, epochs=100, **kwargs)
        elif self.model_type == 'rf':
            self.model.fit(X, y, **kwargs)

    def get_model_type(self):
        return self.model_type

    def __str__(self):
        model_name = self.model_type
        for key, value in self.hyperparameters.items():
            if key == 'layers':
                value_str = "_".join([f"{unit:03d}" for unit in value])
            elif key == 'learning_rate':
                value_str = f"{value:.4f}"
            elif isinstance(value, list):
                value_str = "_".join(map(str, value))
            else:
                value_str = str(value)
            param_name = key[:2]  # Take the first two characters of the hyperparameter name
            model_name += f"_{param_name}{value_str}"
        
        return model_name