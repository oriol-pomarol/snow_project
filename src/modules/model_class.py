import joblib
import json
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.callbacks import Callback
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from config import paths, cfg
from .auxiliary_functions import preprocess_data_lstm

class Model:
    def __init__(self, mode):
        valid_mode = mode.lower() in ['dir_pred', 'err_corr', 'cro_vars', 'data_aug']
        if valid_mode:
            self.mode = mode.lower()
        else:
            raise ValueError(f"Invalid model setup: {valid_mode}.")
        self.model = None
        self.model_type = None
        self.hyperparameters = None

    def set_hps(self, model_type, hyperparameters, epochs=None):
        valid_model_type = model_type.lower() in ['nn', 'rf','lstm']
        if valid_model_type:
            self.model_type = model_type.lower()
        else:
            raise ValueError("Invalid model type.")
        self.hyperparameters = hyperparameters
        self.epochs = epochs
    
    def save_hps(self, path_dir=None):
        if path_dir is None:
            path_dir = paths.temp_data
        hps_mt = {
            'hyperparameters': self.hyperparameters,
            'model_type': self.model_type,
            'epochs': self.epochs
        }
        with open(path_dir / f'{self.mode}_hps.json', 'w') as f:
            json.dump(hps_mt, f)

    def load_hps(self, path_dir=None):
        if path_dir is None:
            path_dir = paths.temp_data
        with open(path_dir / f'{self.mode}_hps.json', 'r') as f:
            hps_mt = json.load(f)
            self.hyperparameters = hps_mt.get('hyperparameters')
            self.model_type = hps_mt.get('model_type')
            self.epochs = hps_mt.get('epochs')

    def create_model(self, input_shape, n_met_vars):
        self.model = None  # Clear any existing model

        if self.model_type == 'nn':
            self.model = keras.Sequential()
            self.model.add(keras.layers.Input(shape=meteo_shape + others_shape))
            for units in self.hyperparameters.get('layers', [128]):
                activation = self.hyperparameters.get('activation', 'relu')
                self.model.add(keras.layers.Dense(units, activation=activation))
            self.model.add(keras.layers.Dense(1, activation='linear'))
            self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.hyperparameters.get('learning_rate', 0.001)),
                               loss='mean_squared_error', metrics=['mean_squared_error'], weighted_metrics=[])
        
        elif self.model_type == 'lstm':
            sequential_input = keras.layers.Input(shape=(cfg.lag, n_met_vars // cfg.lag))
            activation = self.hyperparameters.get('activation', 'relu')
            depth = len(self.hyperparameters.get('layers'))
            x = sequential_input
            for layer, units in enumerate(self.hyperparameters.get('layers', [128])):
                if (depth > 1) and (layer + 1 < depth):
                    x = keras.layers.LSTM(units, return_sequences=True)(x)
                else:
                    x = keras.layers.LSTM(units)(x)
            if input_shape > n_met_vars:
                extra_var_input = keras.layers.Input(shape=(input_shape - n_met_vars,), name='extra_var_input')
                combined_input = keras.layers.Concatenate()([x, extra_var_input])
                x = keras.layers.Dense(units=128, activation=activation)(combined_input)
            output_layer = keras.layers.Dense(1, activation='linear')(x)
            if input_shape > n_met_vars:
                self.model = keras.models.Model(inputs=[sequential_input, extra_var_input], outputs=output_layer)
            else:
                self.model = keras.models.Model(inputs=sequential_input, outputs=output_layer)
            self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.hyperparameters.get('learning_rate', 0.001)),
                               loss='mean_squared_error', metrics=['mean_squared_error'], weighted_metrics=[])
        
        elif self.model_type == 'rf':
            self.model = RandomForestRegressor(n_estimators=200, random_state=10,
                                               max_depth=self.hyperparameters.get('max_depth', None),
                                               max_samples=self.hyperparameters.get('max_samples', None))
            
    def save_model(self, path_dir=None, suffix=''):
        if path_dir is None:
            path_dir = paths.models
        model_name = '_'.join([self.mode, suffix]) if suffix else self.mode
        if self.model_type == 'rf':
            joblib.dump(self.model, path_dir / f'{model_name}.joblib')
        else:
            self.model.save(path_dir / f'{model_name}.h5')
    
    def load_model(self, path_dir=None, suffix=''):
        if path_dir is None:
            path_dir = paths.models
        model_name = '_'.join([self.mode, suffix]) if suffix else self.mode
        if self.model_type == 'rf':
            self.model = joblib.load(path_dir / f'{model_name}.joblib')
        else:
            self.model = keras.models.load_model(path_dir / f'{model_name}.h5')

    def fit(self, X, y, **kwargs):

        # Drop a percentage of the data, unless it is less than 100 samples
        if len(X) > 100 and cfg.drop_data > 0:
            mask = np.random.rand(len(X)) > cfg.drop_data
            X = X[mask]
            y = y[mask]

            # If sample_weight is provided, drop the corresponding samples
            sample_weight = kwargs.get('sample_weight')
            if sample_weight is not None:
                kwargs['sample_weight'] = kwargs['sample_weight'][mask]

        # If it is an lstm model, preprocess the data accordingly
        if self.model_type == 'lstm':
            X = preprocess_data_lstm(X)
        
        # Fit the data with keras if it is a neural network
        if self.model_type in ['nn', 'lstm']:
            callbacks = [SaveModelAtEpoch(self.epochs)] if len(self.epochs) > 1 else []
            history = self.model.fit(X, y, epochs=max(self.epochs), callbacks=callbacks, **kwargs)
            if len(self.epochs) > 1:
                self.model = callbacks[0].get_saved_models()
            return history
        
        # Fit the data with sklearn if it is a random forest
        elif self.model_type == 'rf':
            self.model.fit(X, y, **kwargs)
            return None

    def predict(self, X):

        # If the dataset is empty, return an empty array
        if len(X) == 0:
            return np.array([])

        # If it is an lstm model, preprocess the data accordingly
        if self.model_type == 'lstm':
            X = preprocess_data_lstm(X)

        # Predict the data; if it is not a random forest set the verbose to 0
        if self.model_type == 'rf':
            y_pred = self.model.predict(X)
        else:
            y_pred = self.model.predict(X, verbose=0)
        return y_pred

    def test(self, X, y):

        # Drop a percentage of the data, unless it is less than 100 samples
        if len(X) > 100:
            mask = np.random.rand(len(X)) > cfg.drop_data
            X = X[mask]
            y = y[mask]

        if self.model_type == 'lstm':
            X = preprocess_data_lstm(X)
        if type(self.model) == dict:
            mse = {}
            for epoch, model in self.model.items():
                y_pred = model.predict(X)
                mse[epoch] = mean_squared_error(y, y_pred)
                best_epoch = min(mse, key=mse.get)
            self.model = self.model[best_epoch]
            self.epochs = [best_epoch]
            print(f"Reporting mse for model at epoch {best_epoch}")
            mse = mse[best_epoch]
        else:
            y_pred = self.model.predict(X)
            mse = mean_squared_error(y, y_pred)
        return mse

    def get_model_type(self):
        return self.model_type

    def __str__(self):
        model_name = self.model_type
        for key, value in self.hyperparameters.items():

            # Take the first two characters of the hyperparameter name
            param_name = key[:2]

            # Manually added exceptions for clarity
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
    
# Define a custom callback to save models at specific epochs
class SaveModelAtEpoch(Callback):
    def __init__(self, save_epochs):
        super(SaveModelAtEpoch, self).__init__()
        self.save_epochs = save_epochs
        self.saved_models = {}

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) in self.save_epochs:
            # Clone the model and store it
            model_copy = tf.keras.models.clone_model(self.model)
            model_copy.set_weights(self.model.get_weights())
            self.saved_models[epoch + 1] = model_copy
            print(f"\n\nModel saved at epoch {epoch + 1}\n")

    def get_saved_models(self):
        return self.saved_models