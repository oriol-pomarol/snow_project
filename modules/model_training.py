import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import r2_score
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
    for i, df in enumerate(trng_dfs):
        df = df.loc[df['delta_obs_swe'] != -1 * df['obs_swe'], :]
        trng_dfs[i] = df.dropna()

    for i, df in enumerate(augm_dfs):
        df = df.loc[df['delta_mod_swe'] != -1 * df['mod_swe'], :]
        subset_na = df.columns.difference(['obs_swe', 'delta_obs_swe'])
        augm_dfs[i] = df.dropna(subset=subset_na)
    
    # Obtain the best model for the direct prediction setup
    print('Starting direct prediction training...')
    X_obs = [df.iloc[:int(len(df)*0.8), :-4] for df in trng_dfs]
    y_obs = [df.iloc[:int(len(df)*0.8), -2] for df in trng_dfs]
    X_test = [df.iloc[int(len(df)*0.8):, :-4] for df in trng_dfs]
    y_test = [df.iloc[int(len(df)*0.8):, -2] for df in trng_dfs]
    model_dp = model_selection(X=X_obs, y=y_obs, lag=lag, mode = 'dir_pred')
    plot_pred_vs_true(model=model_dp, X_train=X_obs, y_train=y_obs,
                      X_test=X_test, y_test=y_test, mode='dir_pred')
    print('Direct prediction trained successfully...')

    # Obtain the best model for the error correction setup
    print('Starting error correction training...')
    X_obs = [df.iloc[:int(len(df)*0.8), :-4].join(df.iloc[:, -1]) for df in trng_dfs]
    y_obs = [df.iloc[:int(len(df)*0.8), -2] for df in trng_dfs]
    X_test = [df.iloc[int(len(df)*0.8):, :-4].join(df.iloc[:, -1]) for df in trng_dfs]
    y_test = [df.iloc[int(len(df)*0.8):, -2] for df in trng_dfs]
    model_ec = model_selection(X=X_obs, y=y_obs, lag=lag, mode = 'err_corr')
    plot_pred_vs_true(model=model_ec, X_train=X_obs, y_train=y_obs,
                      X_test=X_test, y_test=y_test, mode='err_corr')
    print('Error correction trained successfully...')

    # Obtain the best model for the data augmentation setup
    print('Starting data augmentation training...')
    X_obs = [df.iloc[:int(len(df)*0.8), :-4] for df in trng_dfs]
    y_obs = [df.iloc[:int(len(df)*0.8), -2] for df in trng_dfs]
    X_aug = [df.iloc[:, :-3] for df in augm_dfs]
    y_aug = [df.iloc[:, -1] for df in augm_dfs]
    X_test = [df.iloc[int(len(df)*0.8):, :-4] for df in trng_dfs]
    y_test = [df.iloc[int(len(df)*0.8):, -2] for df in trng_dfs]
    model_da = model_selection(X=X_obs, y=y_obs, lag=lag, X_aug=X_aug,
                               y_aug=y_aug, mode = 'data_aug')
    plot_pred_vs_true(model=model_da, X_train=X_obs, y_train=y_obs,
                      X_test=X_test, y_test=y_test, mode='data_aug',
                      X_aug=X_aug, y_aug=y_aug)
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

    # Save the train_test split dates as a csv
    train_test_split_dates = [df.index[int(len(df)*0.8)] for df in trng_dfs]
    train_test_split_stations = ['cdp', 'rme', 'sod']
    df_train_test_split = pd.DataFrame({'train_test_split_dates':train_test_split_dates},
                                       index=train_test_split_stations)
    df_train_test_split.to_csv(os.path.join('results', 'train_test_split.csv'))
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

    # Make a random split between training and validation data
    X_train, X_val, y_train, y_val = \
        train_test_split(pd.concat(X), pd.concat(y),
                         test_size=0.2, random_state=10)
    X_train_aug, X_val_aug, y_train_aug, y_val_aug = \
        train_test_split(pd.concat(X_aug), pd.concat(y_aug), 
                         test_size=len(X_val), random_state=10)

    # Define the lengths of the training observed and augmented data 
    len_X_obs_train = len(X_train)
    len_X_aug_train = len(X_train_aug)
    
    # Train each model and calculate the loss on the validation data
    for m, model in enumerate(models):

        model_names.append(str(model))
        print(f'Model {m+1} of {len(models)}.')

        if mode == 'data_aug':
            # Concatenate the observed and augmented data and convert to arrays
            X_train_arr = pd.concat([X_train, X_train_aug]).values
            y_train_arr = pd.concat([y_train, y_train_aug]).values
            X_val_arr = pd.concat([X_val, X_val_aug]).values
            y_val_arr = pd.concat([y_val, y_val_aug]).values

            # Calculate the relative weight of the obs/mod data
            weight_aug = model.hyperparameters.get('rel_weight', 1) * len_X_obs_train / len_X_aug_train
            sample_weight = np.concatenate((np.ones(len_X_obs_train), 
                                            np.full(len_X_aug_train, weight_aug)))
            
        else:
            # Convert the dataframes to arrays
            X_train_arr = X_train.values
            y_train_arr = y_train.values
            X_val_arr = X_val.values
            y_val_arr = y_val.values
            sample_weight = None

        # Create and fit the model, then find the loss
        model.create_model(X_train.shape[1])
        model.fit(X=X_train_arr, y=y_train_arr, X_val=X_val_arr,
                  y_val=y_val_arr, sample_weight=sample_weight)
        loss = model.test(X=X_val_arr, y=y_val_arr)
        losses[m] = loss

    # Select the best model
    best_model = models[np.argmin(losses)]

    # If in data augmentation, tune the relative weight of the obs/mod data
    if mode == 'data_aug':
        models_rw = []
        losses_rw = np.zeros(len(rel_weight_vals))
        model_rw_1 = best_model
        X_train_arr = pd.concat([X_train, X_train_aug]).values
        y_train_arr = pd.concat([y_train, y_train_aug]).values
        X_val_arr = pd.concat([X_val, X_val_aug]).values
        y_val_arr = pd.concat([y_val, y_val_aug]).values

        for weight_idx, rel_weight in enumerate(rel_weight_vals):
            model_names.append(str(best_model) + f'_rw_{rel_weight}')
            if rel_weight == 1:
                best_model = model_rw_1
                loss = np.min(losses)
            else:
                print(f'Relative weight {weight_idx+1} of {len(models)}.')
                weight_aug = rel_weight * len_X_obs_train / len_X_aug_train
                sample_weight = np.concatenate((np.ones(len_X_obs_train),
                                                np.full(len_X_aug_train, weight_aug)))
                best_model.create_model(X_train.shape[1])
                best_model.fit(X=X_train_arr, y=y_train_arr, 
                               sample_weight=sample_weight)
                loss = best_model.test(X=X_val_arr, y=y_val_arr)
            
            models_rw.append(best_model)
            losses_rw[weight_idx] = loss

        # Select the best model
        best_rw = rel_weight_vals[np.argmin(losses_rw)]
        best_model = models_rw[np.argmin(losses_rw)]
        losses = np.append(losses, losses_rw)

    # Save the model hyperparameters and their losses as a csv
    df_losses = pd.DataFrame({'MSE (mean)':losses,'HP':model_names})
    df_losses.set_index('HP', inplace=True)
    df_losses.to_csv(os.path.join('results', f'model_losses_{mode}.csv'))

    # Train the best model on all train/val data
    X_train, y_train = pd.concat(X), pd.concat(y)
    X_train_aug, y_train_aug = pd.concat(X_aug), pd.concat(y_aug)
    
    if mode == 'data_aug':
        # Define the lengths of the training observed and augmented data 
        len_X_obs_train = len(X_train)
        len_X_aug_train = len(X_train_aug)

        # Concatenate the observed and augmented data and convert to arrays
        X_train_arr = pd.concat([X_train, X_train_aug]).values
        y_train_arr = pd.concat([y_train, y_train_aug]).values

        # Calculate the relative weight of the obs/mod data
        weight_aug = model.hyperparameters.get('rel_weight', 1) * len_X_obs_train / len_X_aug_train
        sample_weight = np.concatenate((np.ones(len_X_obs_train), 
                                        np.full(len_X_aug_train, weight_aug)))
        
    else:
        # Convert the dataframes to arrays
        X_train_arr = X_train.values
        y_train_arr = y_train.values
        sample_weight = None

    best_model.create_model(X[0].shape[1])
    history = best_model.fit(X=X_train_arr, y=y_train_arr,
                             sample_weight=sample_weight)

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

###############################################################################

def plot_pred_vs_true(model, X_train, y_train, X_test, y_test, mode, X_aug=None, y_aug=None):
    X_train_arr = pd.concat(X_train).values
    y_train_arr = pd.concat(y_train).values
    X_test_arr = pd.concat(X_test).values
    y_test_arr = pd.concat(y_test).values
    
    # Predict values for training and test data
    y_train_pred = model.predict(X_train_arr).ravel()
    y_test_pred = model.predict(X_test_arr).ravel()

    if mode == 'data_aug':
        X_aug_arr = pd.concat(X_aug).values
        y_aug_arr = pd.concat(y_aug).values
        y_aug_pred = model.predict(X_aug_arr).ravel()

    # Create scatter plot for training data
    fig = plt.figure(figsize=(12, 7))

    white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
        (0, '#ffffff'),
        (1e-20, '#440053'),
        (0.2, '#404388'),
        (0.4, '#2a788e'),
        (0.6, '#21a784'),
        (0.8, '#78d151'),
        (1, '#fde624'),
    ], N=256)

    ax = fig.add_subplot(1, 2, 1 if mode == 'data_aug' else 1)
    ax.set_aspect('equal', adjustable='box')
    min_val, max_val = np.percentile(np.concatenate([y_train_arr, y_train_pred]), [1, 99])
    density = ax.hist2d(y_train_arr, y_train_pred, bins=range(int(min_val), int(max_val) + 1), cmap=white_viridis)
    fig.colorbar(density[3], ax=ax, label='Number of points per bin')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Train Data')
    plt.plot([min_val, max_val], [min_val, max_val], 'k-', lw=1)

    # Calculate R-squared value and add it to the plot
    r2 = r2_score(y_train_arr, y_train_pred)
    ax.text(0.05, 0.95, f'R-squared = {r2:.2f}', transform=ax.transAxes, fontsize=14,
            verticalalignment='top')

    if mode == 'data_aug':
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.set_aspect('equal', adjustable='box')
        min_val, max_val = np.percentile(np.concatenate([y_aug_arr, y_aug_pred]), [1, 99])
        density = ax2.hist2d(y_aug_arr, y_aug_pred, bins=range(int(min_val), int(max_val) + 1), cmap=white_viridis)
        fig.colorbar(density[3], ax=ax2, label='Number of points per bin')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title('Augmented Data')
        plt.plot([min_val, max_val], [min_val, max_val], 'k-', lw=1)

        # Calculate R-squared value and add it to the plot
        r2 = r2_score(y_aug_arr, y_aug_pred)
        ax2.text(0.05, 0.95, f'R-squared = {r2:.2f}', transform=ax2.transAxes, fontsize=14,
                verticalalignment='top')

    plt.tight_layout()
    plt.savefig(os.path.join('results', f'pred_vs_true_{mode}.png'))

    # Create scatter plot for test data
    fig_test = plt.figure(figsize=(12, 7))

    ax_test = fig_test.add_subplot(1, 1, 1)
    ax_test.set_aspect('equal', adjustable='box')
    min_val_test, max_val_test = np.percentile(np.concatenate([y_test_arr, y_test_pred]), [1, 99])
    density_test = ax_test.hist2d(y_test_arr, y_test_pred, bins=range(int(min_val_test), int(max_val_test) + 1), cmap=white_viridis)
    fig_test.colorbar(density_test[3], ax=ax_test, label='Number of points per bin')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Test Data')
    plt.plot([min_val_test, max_val_test], [min_val_test, max_val_test], 'k-', lw=1)

    # Calculate R-squared value and add it to the plot
    r2_test = r2_score(y_test_arr, y_test_pred)
    ax_test.text(0.05, 0.95, f'R-squared = {r2_test:.2f}', transform=ax_test.transAxes, fontsize=14,
        verticalalignment='top')

    plt.tight_layout()
    plt.savefig(os.path.join('results', 'pred_vs_true_test.png'))

    # Save the true and predicted values as csv
    train_df = pd.DataFrame({'TrueValues': y_train_arr, 'PredictedValues': y_train_pred})
    train_df.to_csv(os.path.join('results', f'pred_vs_true_{mode}.csv'), index=False)

    if mode == 'data_aug':
        aug_df = pd.DataFrame({'TrueValues': y_aug_arr, 'PredictedValues': y_aug_pred})
        aug_df.to_csv(os.path.join('results', f'pred_vs_true_{mode}_aug.csv'), index=False)

    # Save the true and predicted values as csv
    test_df = pd.DataFrame({'TrueValues': y_test_arr, 'PredictedValues': y_test_pred})
    test_df.to_csv(os.path.join('results', 'pred_vs_true_test.csv'), index=False)

###############################################################################

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
            X = preprocess_data_lstm(X, self.lag)
            X_val = preprocess_data_lstm(X_val, self.lag)      
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
            X = preprocess_data_lstm(X, self.lag)
        if self.model_type == 'lstm' and self.mode == 'err_corr':
            y_pred = self.model.predict([X,X_mod])
        else:
            y_pred = self.model.predict(X)
        return y_pred
        
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