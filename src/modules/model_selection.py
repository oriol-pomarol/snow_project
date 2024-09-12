import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from config import cfg, paths
from model_class import Model

def model_selection(X, y, X_aug=[], y_aug=[], mode=''):
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
            model = Model(mode, 'rf', cfg.lag)
            model.set_hyperparameters(max_depth=max_depth,
                                      max_samples=max_samples)
            models.append(model)

    # Initialize a NN model for each combination of HP
    for layers in layers_nn_vals:
        for learning_rate in learning_rate_vals:
            for l2_reg in l2_reg_vals:
                model = Model(mode, 'nn', cfg.lag)
                model.set_hyperparameters(layers=layers,
                                          learning_rate=learning_rate,
                                          l2_reg=l2_reg)
                models.append(model)

    # Initialize a LSTM model for each combination of HP
    if cfg.lag > 0:
        for layers in layers_lstm_vals:
            for learning_rate in learning_rate_vals:
                for l2_reg in l2_reg_vals:
                    model = Model(mode, 'lstm', cfg.lag)
                    model.set_hyperparameters(layers=layers,
                                              learning_rate=learning_rate,
                                              l2_reg=l2_reg)
                    models.append(model)

    # Initialize losses and model names for model validation
    if cfg.temporal_split:
        losses = np.zeros((len(models), 1))
    else:
        losses = np.zeros((len(models), len(X)))
    model_names = []

    # Initialize training and validation datasets
    train_val_splits = []

    if cfg.temporal_split:
        # Make a random split between training and validation data
        X_train, X_val, y_train, y_val = \
            train_test_split(pd.concat(X), pd.concat(y),
                            test_size=0.2, random_state=10)
        train_val_splits.append([X_train, X_val, y_train, y_val])
        if mode == 'data_aug':
            X_train_aug, X_val_aug, y_train_aug, y_val_aug = \
                train_test_split(pd.concat(X_aug), pd.concat(y_aug), 
                                test_size=len(X_val), random_state=10)
            train_val_splits[-1] += [X_train_aug, X_val_aug, y_train_aug, y_val_aug]
        
    else:
        for i in range(len(X)):
            print(f'Train/val split {i+1} of {len(X)}.')
            X_train = pd.concat([X[j] for j in range(len(X)) if j!=i])
            y_train = pd.concat([y[j] for j in range(len(y)) if j!=i])
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                              test_size=0.2, random_state=10)
            train_val_splits.append([X_train, X_val, y_train, y_val])
            if mode == 'data_aug':
                X_train_aug = pd.concat([X_train, pd.concat(X_aug)])
                y_train_aug = pd.concat([y_train, pd.concat(y_aug)])
                X_train_aug, X_val_aug, y_train_aug, y_val_aug = train_test_split(X_train_aug, y_train_aug,
                                                                                test_size=0.2, random_state=10)
                train_val_splits[-1] += [X_train_aug, X_val_aug, y_train_aug, y_val_aug]

                
            else:
                sample_weight = None
    
    # Train each model and calculate the loss on the validation data
    for m, model in enumerate(models):

        model_names.append(str(model))
        print(f'Model {m+1} of {len(models)}.')

        for i, train_val_split in enumerate(train_val_splits):

            if mode == 'data_aug':

                # Unpack the training and validation data
                X_train, X_val, y_train, y_val, X_train_aug, X_val_aug, y_train_aug, y_val_aug = train_val_split

                # Define the lengths of the training observed and augmented data 
                len_X_obs_train = len(X_train)
                len_X_aug_train = len(X_train_aug)

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
                # Unpack the training and validation data
                X_train, X_val, y_train, y_val = train_val_split

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
            losses[m, i] = loss

    # Select the best model
    mean_loss = np.mean(losses, axis=1)
    best_model = models[np.argmin(mean_loss)]

    # Save the model hyperparameters and their losses as a csv   
    if losses.shape[1] == 1:
        df_losses = pd.DataFrame({'MSE': losses[:, 0], 'HP': model_names})
    else:
        data = {f'MSE (Split {i+1})': losses[:, i] for i in range(losses.shape[1])}
        data.update({'MSE (mean)': mean_loss, 'HP': model_names})
        df_losses = pd.DataFrame(data)
    df_losses.set_index('HP', inplace=True)
    df_losses.to_csv(paths.outputs / f'model_losses_{mode}.csv')

    return best_model