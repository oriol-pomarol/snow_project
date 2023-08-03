import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import os
import joblib

def model_training(dfs_obs, dfs_meteo_agg, dfs_mod):
    # Choose what dfs can be used for testing and what only for observations
    dfs_test_idx = [1,2,3,5,6,8,9]
    dfs_obs_train_idx = [0,4,7]

    # Direct prediction
    X = pd.concat([dfs_meteo_agg[j].loc[dfs_obs[j].index] for j in dfs_obs_train_idx])
    y = pd.concat([dfs_obs[j] for j in dfs_obs_train_idx])
    train_model(X=X, y=y, name='dir_pred')

    # Error correction
    X = pd.concat([pd.concat([dfs_meteo_agg[j].loc[dfs_obs[j].index],
                            dfs_mod[j].loc[dfs_obs[j].index]], axis=1) for j in dfs_obs_train_idx])
    y = pd.concat([dfs_obs[j] for j in dfs_obs_train_idx])
    train_model(X=X, y=y, name='err_corr')

    # # Data augmentation
    # for i in dfs_test_idx:
    #     X_obs = pd.concat([dfs_meteo_agg[j].loc[dfs_obs[j].index] for j in dfs_obs_train_idx])
    #     X_mod = pd.concat([dfs_meteo_agg[j].loc[dfs_mod[j].index] for j in dfs_test_idx if j!=i])
    #     y = pd.concat([pd.concat([dfs_obs[j] for j in dfs_obs_train_idx]),
    #                   pd.concat([dfs_mod[j] for j in dfs_test_idx if j!=i])])
    #     weight_train_mod = weight_mod * len(X_obs) / len(X_mod)
    #     sample_weight = np.concatenate((np.ones(len(X_obs)), np.full(len(X_mod), weight_train_mod)))
    #     train_model(X=pd.concat([X_obs,X_mod]), y=y, sample_weight=sample_weight, name=f'data_aug_{i}')

####################################################################################
# EXTRA FUNCTIONS
####################################################################################

def train_model(X, y, name, sample_weight=None):
    # Split between train and test set
    X_train, X_val, y_train, y_val = train_test_split(X.values, y.values,
                                                    test_size=0.2, random_state=10)

    # Define the models
    rforest = RandomForestRegressor()
    nnetwork = keras.Sequential([
        keras.layers.Dense(20, activation='relu', input_shape=(X_train.shape[1],)),
        keras.layers.Dense(20, activation='relu'),
        keras.layers.Dense(1)  # Output layer with a single unit for regression
    ])

    # Compile the nnetwork
    nnetwork.compile(optimizer='adam', loss='mean_squared_error')

    # Train the models
    rforest.fit(X_train, y_train.ravel(), sample_weight=sample_weight)
    history = nnetwork.fit(X_train, y_train, epochs=10, batch_size=32, 
                           validation_data=(X_val, y_val), sample_weight=sample_weight)
    
    # Plot the MSE history of the training
    plt.figure()
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    # plt.yscale('log')
    plt.legend()
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.savefig(os.path.join('results',f'train_history_{name}.png'))

    # Save the models
    joblib.dump(rforest, os.path.join('results', 'models', f'rf_{name}.joblib'))
    nnetwork.save(os.path.join('results', 'models', f'nn_{name}.h5'))

    return