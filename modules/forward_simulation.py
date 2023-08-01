import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tensorflow import keras
import os

def forward_simulation(obs, mod, meteo_agg, mod_delta_swe_all, model_names, station_name):
    # Initialize a vector for the predicted and observed SWE
    pred_swe_arr = np.zeros((len(model_names), len(meteo_agg)))
    mse_swe_list = []

    # Make the forward simulation
    for i, model_name in enumerate(model_names):
        # Find the trained model
        if 'rf' in model_name:
            model = joblib.load(os.path.join('results', 'models', model_name))
        if 'nn' in model_name:
            model = keras.models.load_model(os.path.join('results', 'models', model_name))

        # Define the X according to the model
        if ('dir_pred' in model_name) or ('data_aug' in model_name):
            fwd_X = meteo_agg
        elif 'err_corr' in model_name:
            fwd_X = pd.concat(meteo_agg, mod_delta_swe_all)

        for j in range(len(pred_swe_arr)):
            pred_y = model.predict(fwd_X.values[[j-1]])
            pred_swe_arr[i,j] = max(pred_swe_arr[i,j-1] + pred_y, 0)
        
        # Find the MSE and store it in the list
        pred_obs = pred_swe_arr[i][np.isin(fwd_X.index, obs.index)]
        mse_swe = mean_squared_error(obs.values, pred_obs)
        mse_swe_list.append(mse_swe)


    # Save the MSE list
    with open(os.path.join('results', f'{station_name}_mse.csv'), 'w') as file:
        # Convert the list to a string representation (comma-separated values)
        list_as_string = ','.join(str(item) for item in mse_swe_list)
        # Write the string representation to the file
        file.write(list_as_string)

    # Plot the results
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, model_name in enumerate(model_names):
        ax.plot(fwd_X.index, pred_swe_arr[i,:], label=model_name)
    ax.plot(obs.index, obs.values, label='Observed SWE')
    ax.plot(mod.index, mod.values, label="Modelled SWE")
    ax.set_xlabel('Date')
    ax.set_ylabel('SWE')
    ax.legend()
    plt.savefig(os.path.join('results', f'{station_name}_fwd_sim.png'))

    return