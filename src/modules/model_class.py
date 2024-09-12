import joblib
from tensorflow import keras
from keras.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from config import paths
from .auxiliary_functions import preprocess_data_lstm

class Model:
    def __init__(self, mode, model_type, lag):
        valid_model_type = model_type.lower() in ['nn', 'rf','lstm'] 
        valid_mode = mode.lower() in ['dir_pred', 'err_corr', 'data_aug']
        if valid_model_type and valid_mode:
            self.model_type = model_type.lower()
            self.mode = mode.lower()
            self.lag = lag
            self.model = None
            if (self.model_type == 'nn') or (self.model_type == 'lstm'):
                self.epochs = 100
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

    def fit(self, X, y, X_val=None, y_val=None, **kwargs):
        if self.model_type == 'lstm':
            if self.mode == 'err_corr':
                X_mod = X[:,-1]
                X = X[:,:-1]
                X_val_mod = X_val[:,-1]
                X_val = X_val[:,:-1]
            X = preprocess_data_lstm(X)
            X_val = preprocess_data_lstm(X_val)      
        if self.model_type in ['nn', 'lstm']:
            # Define early stopping callback
            callbacks = []
            if (X_val is not None) and (y_val is not None):
                callbacks = [EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)]
            if self.model_type == 'lstm' and self.mode == 'err_corr':
                history = self.model.fit([X,X_mod], y, epochs=self.epochs, validation_data=([X_val,X_val_mod], y_val),
                                         callbacks=callbacks, **kwargs)
            else:
                history = self.model.fit(X, y, epochs=self.epochs, validation_data=(X_val, y_val), callbacks=callbacks, **kwargs)
            self.epochs = len(history.history['loss'])
            return history
        elif self.model_type == 'rf':
            self.model.fit(X, y.ravel(), **kwargs)
            return None

    def predict(self, X):
        if self.model_type == 'lstm':
            if self.mode == 'err_corr':
                X_mod = X[:,-1]
                X = X[:,:-1]
            X = preprocess_data_lstm(X)
        if self.model_type == 'lstm' and self.mode == 'err_corr':
            y_pred = self.model.predict([X,X_mod])
        else:
            y_pred = self.model.predict(X)
        return y_pred
    
    def save(self):
        if self.model_type == 'rf':
            joblib.dump(self.model, paths.models / f'{self.mode}.joblib')
        else:
            self.model.save(paths.models / f'{self.mode}.h5')
        
    def test(self, X, y):
        if self.model_type == 'lstm':
            if self.mode == 'err_corr':
                X_mod = X[:,-1]
                X = X[:,:-1]
            X = preprocess_data_lstm(X)
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