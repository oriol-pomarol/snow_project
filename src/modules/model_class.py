import joblib
import json
from tensorflow import keras
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from config import paths, cfg
from .auxiliary_functions import preprocess_data_lstm

class Model:
    def __init__(self, mode):
        valid_mode = mode.lower() in ['dir_pred', 'err_corr', 'data_aug']
        if valid_mode:
            self.mode = mode.lower()
        else:
            raise ValueError("Invalid model setup.")
        self.model = None
        self.model_type = None
        self.hyperparameters = None

    def set_hps(self, model_type, hyperparameters):
        valid_model_type = model_type.lower() in ['nn', 'rf','lstm']
        if valid_model_type:
            self.model_type = model_type.lower()
        else:
            raise ValueError("Invalid model type.")
        self.hyperparameters = hyperparameters
    
    def save_hps(self, path_dir=None):
        if path_dir is None:
            path_dir = paths.temp_data
        hps_mt = {
            'hyperparameters': self.hyperparameters,
            'model_type': self.model_type
        }
        with open(path_dir / f'{self.mode}_hps.json', 'w') as f:
            json.dump(hps_mt, f)

    def load_hps(self, path_dir=None):
        if path_dir is None:
            path_dir = paths.temp_data
        with open(path_dir / f'{self.mode}_hps.json', 'r') as f:
            hps_mt = json.load(f)
            self.hyperparameters = hps_mt.get('hyperparameters', {})
            self.model_type = hps_mt.get('model_type')

    def create_model(self, input_shape, crocus_shape=0):
        self.model = None  # Clear any existing model

        if (self.model_type == 'nn') or (self.model_type == 'lstm'):
            self.epochs = 100

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
            sequential_input = keras.layers.Input(shape=(cfg.lag, (input_shape-1*crocus_shape*(self.mode=='err_corr')) // cfg.lag))
            activation = self.hyperparameters.get('activation', 'relu')
            depth = len(self.hyperparameters.get('layers'))
            x = sequential_input
            for layer, units in enumerate(self.hyperparameters.get('layers', [128])):
                if (depth > 1) and (layer + 1 < depth):
                    x = keras.layers.LSTM(units, return_sequences=True)(x)
                else:
                    x = keras.layers.LSTM(units)(x)
            if self.mode == 'err_corr' and crocus_shape > 0:
                extra_var_input = keras.layers.Input(shape=(crocus_shape,), name='extra_var_input')
                combined_input = keras.layers.Concatenate()([x, extra_var_input])
                x = keras.layers.Dense(units=128, activation=activation)(combined_input)
            output_layer = keras.layers.Dense(1, activation='linear')(x)
            if self.mode == 'err_corr' and crocus_shape > 0:
                self.model = keras.models.Model(inputs=[sequential_input, extra_var_input], outputs=output_layer)
            else:
                self.model = keras.models.Model(inputs=sequential_input, outputs=output_layer)
            self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.hyperparameters.get('learning_rate', 0.001)),
                               loss='mean_squared_error', metrics=['mean_squared_error'], weighted_metrics=[])
        elif self.model_type == 'rf':
            self.model = RandomForestRegressor(n_estimators=200, random_state=10,
                                               max_depth=self.hyperparameters.get('max_depth', None),
                                               max_samples=self.hyperparameters.get('max_samples', None))
            
    def save_model(self, path_dir=None):
        if path_dir is None:
            path_dir = paths.models
        if self.model_type == 'rf':
            joblib.dump(self.model, path_dir/ f'{self.mode}.joblib')
        else:
            self.model.save(path_dir / f'{self.mode}.h5')

    def load_model(self, path_dir=None):
        if path_dir is None:
            path_dir = paths.models
        if self.model_type == 'rf':
            self.model = joblib.load(path_dir / f'{self.mode}.joblib')
        else:
            self.model = keras.models.load_model(path_dir / f'{self.mode}.h5')

    def fit(self, X, y, **kwargs):

        # If it is an lstm model, preprocess the data accordingly
        if self.model_type == 'lstm':
            X = preprocess_data_lstm(X, mode=self.mode)
        
        # Fit the data with keras if it is a neural network
        if self.model_type in ['nn', 'lstm']:
            history = self.model.fit(X, y, epochs=self.epochs, **kwargs)
            return history
        
        # Fit the data with sklearn if it is a random forest
        elif self.model_type == 'rf':
            self.model.fit(X, y, **kwargs)
            return None

    def predict(self, X):
        if self.model_type == 'lstm':
            X = preprocess_data_lstm(X, mode=self.mode)
        y_pred = self.model.predict(X)
        return y_pred

    def test(self, X, y):
        if self.model_type == 'lstm':
            X = preprocess_data_lstm(X, mode=self.mode)
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