import joblib
import json
import time
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.callbacks import Callback
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from config import paths, cfg
from .auxiliary_functions import preprocess_data_lstm, drop_samples

class Model:
    def __init__(self, mode):
        """
        At initialization, set parameters to None and assign the mode.

        Parameters:
        mode (str): The simulation mode (e.g., 'dir_pred', 'post_prc').
        """
        self.mode = mode.lower()
        self.model = None
        self.model_type = None
        self.hyperparameters = None
        self.rel_weight = None
        self.scaler = None
        self.best_epochs = []

    def set_hps(self, model_type, hyperparameters, epochs=None):
        """
        Set the hyperparameters for the model.

        Parameters:
        model_type (str): The type of the model ('nn', 'lstm' or 'rf').
        hyperparameters (dict): The hyperparameters for the model.
        epochs (list, optional): The epochs at which to save the model.
        """
        self.model_type = model_type.lower()
        self.hyperparameters = hyperparameters
        self.epochs = epochs
    
    def save_hps(self, path_dir=None):
        """
        Save the hyperparameters to a JSON file.

        Parameters:
        path_dir (Path, optional): The directory to save the hyperparameters file.
        """
        if path_dir is None:
            path_dir = paths.temp
        hps_mt = {
            'hyperparameters': self.hyperparameters,
            'model_type': self.model_type,
            'epochs': self.epochs,
            'rel_weight': self.rel_weight,
        }
        with open(path_dir / f'{self.mode}_hps.json', 'w') as f:
            json.dump(hps_mt, f)

    def load_hps(self, path_dir=None):
        """
        Load the hyperparameters from a JSON file.

        Parameters:
        path_dir (Path, optional): The directory to load the hyperparameters file from.
        """
        if path_dir is None:
            path_dir = paths.temp
        with open(path_dir / f'{self.mode}_hps.json', 'r') as f:
            hps_mt = json.load(f)
            self.hyperparameters = hps_mt.get('hyperparameters')
            self.model_type = hps_mt.get('model_type')
            self.epochs = hps_mt.get('epochs')
            self.rel_weight = hps_mt.get('rel_weight')

    def create_model(self, input_shape, n_met_vars):
        """
        Create the model based on the stored type and hyperparameters.

        Parameters:
        input_shape (tuple): The shape of the input data.
        n_met_vars (int): The number of meteorological variables.
        """
        # Clear any existing model
        self.model = None 

        # If the model type is a neural network, create a keras model accordingly
        if self.model_type == 'nn':
            self.model = keras.Sequential()
            self.model.add(keras.layers.Input(shape=input_shape))
            for units in self.hyperparameters.get('layers', [128]):
                activation = self.hyperparameters.get('activation', 'relu')
                self.model.add(keras.layers.Dense(units, activation=activation))
            self.model.add(keras.layers.Dense(1, activation='linear'))
            self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.hyperparameters.get('learning_rate', 0.001)),
                               loss='mean_squared_error', metrics=['mean_squared_error'], weighted_metrics=[])
        
        # If the model type is an lstm, create a keras model accordingly
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
        
        # If the model type is a random forest, create an sklearn model accordingly
        elif self.model_type == 'rf':
            self.model = RandomForestRegressor(n_estimators=200, random_state=10,
                                               max_depth=self.hyperparameters.get('max_depth', None),
                                               max_samples=self.hyperparameters.get('max_samples', None))

    def save_model(self, suffix=''):
        """
        Save the model to a joblib or h5 file.

        Parameters:
        suffix (str, optional): The suffix to add to the model name.
        """
        # Include the suffix (if provided) to the mode to generate the model name
        model_name = '_'.join([self.mode, suffix]) if suffix else self.mode

        # Save the model as a joblib file if it is a random forest
        if self.model_type == 'rf':
            joblib.dump(self.model, paths.models / f'{model_name}.joblib')
        
        # Save the model as a h5 file if it is a neural network
        else:
            self.model.save(paths.models / f'{model_name}.h5')
            joblib.dump(self.scaler, paths.temp / f'{model_name}_scaler.joblib')

    
    def load_model(self, suffix=''):
        """
        Load the model from a file.

        Parameters:
        suffix (str, optional): The suffix to add to the model name.
        """
        # Include the suffix (if provided) to the mode to generate the model name
        model_name = '_'.join([self.mode, suffix]) if suffix else self.mode

        # Load the model from a joblib file if it is a random forest
        if self.model_type == 'rf':
            self.model = joblib.load(paths.models / f'{model_name}.joblib')
        
        # Load the model from a h5 file if it is a neural network
        else:
            self.model = keras.models.load_model(paths.models / f'{model_name}.h5')
            self.scaler = joblib.load(paths.temp / f'{model_name}_scaler.joblib')


    def fit(self, X, y, save_train_history=False, **kwargs):
        """
        Fit the model to the training data.

        Parameters:
        X (DataFrame): The input features.
        y (Series): The target values.
        save_train_history (bool, optional): Whether to save the training history.
        kwargs: Additional arguments for the fit method.
        """
        # Drop a percentage of the data if specified
        if cfg.drop_data > 0:
            sample_weight = kwargs.get("sample_weight", None)
            X, y, sample_weight = drop_samples([X, y, sample_weight], cfg.drop_data)
            kwargs["sample_weight"] = sample_weight

        # Start a timer to measure the training time
        start_train_time = time.time()
        
        # Fit the data with keras if it is a neural network
        if self.model_type in ['nn', 'lstm']:

            # Normalize the input data
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

            # If it is an lstm model, preprocess the data accordingly
            if self.model_type == 'lstm':
                X = preprocess_data_lstm(X)

            # Set a random seed for tensorflow
            tf.random.set_seed(10)
            
            # Fit the model, saving it at the specified epochs
            callbacks = [SaveModelAtEpoch(self.epochs)] if len(self.epochs) > 1 else []
            history = self.model.fit(X, y.squeeze(), epochs=max(self.epochs), verbose=2,
                                     callbacks=callbacks, **kwargs)

            # Define the model as all the saved models (if more than one)
            if len(self.epochs) > 1:
                self.model = callbacks[0].get_saved_models()

            # If specified, save the training history...
            if save_train_history:
                # 1) as a csv
                history_df = pd.DataFrame(history.history)
                history_df.to_csv(paths.temp / f'train_history_{self.mode}.csv')
                                
                # 2) as a plot
                plt.figure()
                plt.plot(history.history['loss'], label='loss')
                plt.legend()
                plt.yscale('log')
                plt.xlabel('Epoch')
                plt.ylabel('MSE')
                plt.savefig(paths.figures / f'train_history_{self.mode}.png')
        
        # Fit the data with sklearn if it is a random forest
        elif self.model_type == 'rf':
            self.model.fit(X, y.squeeze(), **kwargs)

        # Print the training time
        print(f"Training time: {time.time() - start_train_time:.2f} seconds.")

        return

    def predict(self, X):
        """
        Predict the target values for the input features.

        Parameters:
        X (DataFrame): The input features.

        Returns:
        np.array: The predicted target values.
        """
        # If the dataset is empty, return an empty array
        if len(X) == 0:
            return np.array([])

        # Predict the data directly if it is a random forest
        if self.model_type == 'rf':
            y_pred = self.model.predict(X)

        else:
            # Normalize the input data
            X_scaled = self.scaler.transform(X)
            X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

            # If it is an lstm model, preprocess the data accordingly
            if self.model_type == 'lstm':
                X = preprocess_data_lstm(X)
            
            # Predict the data with verbose=0
            y_pred = self.model.predict(X, verbose=0)

        return y_pred

    def test(self, X, y):
        """
        Test the model on the input features and target values.

        Parameters:
        X (DataFrame): The input features.
        y (Series): The target values.

        Returns:
        float: The mean squared error of the predictions.
        """
        # Drop a percentage of the data if specified
        if cfg.drop_data > 0:
            X, y = drop_samples([X, y], cfg.drop_data)

        # Normalize the input data
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
            X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

        # If it is an lstm model, preprocess the data accordingly
        if self.model_type == 'lstm':
            X = preprocess_data_lstm(X)

        if type(self.model) == dict:
            mse = {}
            for epoch, model in self.model.items():
                y_pred = model.predict(X)
                mse[epoch] = mean_squared_error(y, y_pred)
            best_epoch = min(mse, key=mse.get)
            self.best_epochs.append(best_epoch)
            print(f"Reporting mse for model at epoch {best_epoch}")
            mse = mse[best_epoch]
        else:
            y_pred = self.model.predict(X)
            mse = mean_squared_error(y, y_pred)
        return mse

    def get_model_type(self):
        """
        Get the type of the model.

        Returns:
        str: The type of the model.
        """
        return self.model_type

    def __str__(self):
        """
        Get the string representation of the model type and hyperparameters.

        Returns:
        str: The string representation of the model.
        """
        # Initialize the model name with the model type
        model_name = self.model_type

        # Add the hyperparameters to the model name
        for key, value in self.hyperparameters.items():

            # Take the first two characters of the hyperparameter name
            param_name = key[:2]

            # Manually added exceptions for clarity
            if key == 'layers':
                value_str = "_".join([f"{unit:03d}" for unit in value])
                param_name = 'la'
            elif key == 'learning_rate':
                value_str = f"{value:.0e}".replace("-", "_")
                param_name = 'lr'
            elif key == 'l2_reg':
                value_str = f"{value:.0e}".replace("-", "_")
                param_name = 'rg'
            else:
                value_str = str(value)

            # Add the hyperparameters to the model name
            model_name += f"_{param_name}_{value_str}"

        # Add the relative weight to the model name, if existant
        if self.rel_weight is not None:
            model_name += f"_rw_{self.rel_weight:.0e}".replace("-", "_")
            
        return model_name

###############################################################################

# Define a custom callback to save models at specific epochs
class SaveModelAtEpoch(Callback):
    def __init__(self, save_epochs):
        """
        Initialize the SaveModelAtEpoch callback.

        Parameters:
        save_epochs (list): The epochs at which to save the model.
        """
        super(SaveModelAtEpoch, self).__init__()
        self.save_epochs = save_epochs
        self.saved_models = {}

    def on_epoch_end(self, epoch, logs=None):
        """
        Save the model at the specified epochs.

        Parameters:
        epoch (int): The current epoch.
        logs (dict): The logs of the model.
        """
        if (epoch + 1) in self.save_epochs:
            # Clone the model and store it
            model_copy = tf.keras.models.clone_model(self.model)
            model_copy.set_weights(self.model.get_weights())
            self.saved_models[epoch + 1] = model_copy
            print(f"\n\nModel saved at epoch {epoch + 1}\n")

    def get_saved_models(self):
        """
        Get the saved models.

        Returns:
        dict: The saved models.
        """
        return self.saved_models