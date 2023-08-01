import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import os
import joblib

def model_training(X, y, name, sample_weight=None):
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