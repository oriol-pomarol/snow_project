import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from tensorflow import keras
import os

def forward_simulation(obs, mod, meteo_agg, mod_delta_swe_all, model_names):
    # Initialize a vector for the predicted and observed SWE
    pred_swe = np.zeros((len(model_names), len(meteo_agg)))

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

        for j in range(len(pred_swe)):
            pred_y = model.predict(fwd_X.values[[j-1]])
            pred_swe[i,j] = max(pred_swe[i,j-1] + pred_y, 0)

    # Plot the results
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, model_name in enumerate(model_names):
        ax.plot(fwd_X.index, pred_swe[i,:], label=model_name)
    ax.plot(obs.index, obs.values, label='Observed SWE')
    ax.plot(mod.index, mod.values, label="Modelled SWE")
    ax.set_xlabel('Date')
    ax.set_ylabel('SWE')
    ax.legend()
    plt.savefig(os.path.join('results', 'forward_simulation.png'))
    return