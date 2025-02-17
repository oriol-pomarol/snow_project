import numpy as np
import pandas as pd
import shap
import fastdtw
import matplotlib.pyplot as plt
from config import cfg, paths
from .model_class import Model
from .auxiliary_functions import (
    load_processed_data,
    get_split_info,
    temporal_test_split,
    mask_measurements_by_year,
)

def forward_simulation():

    # Load the processed data
    dict_dfs = load_processed_data()

    # Clean the dataframes based on the predictor columns
    for station, df in dict_dfs.items():
        predictor_cols = df.filter(regex='^(met_|cro_)').columns
        df_stn_clean = df.dropna(subset=predictor_cols)
        dict_dfs[station] = df_stn_clean
    
    # Load the models for every mode
    dict_models = {}
    for mode in cfg.modes().keys():

        # Get the number of splits and suffix
        n_splits, suffix = get_split_info(mode)

        # Load the models for each cross validation split
        model_list = []
        for s in range(n_splits):
            model = Model(mode)
            model.load_hps()
            model.load_model(suffix=f"{suffix}_{s}" if suffix else '')
            model_list.append(model)
        dict_models[mode] = model_list

    # Load the split dates and convert them to datetime
    df_split_dates = pd.read_csv(paths.temp / 'split_dates.csv', index_col=[0, 1])
    df_split_dates = df_split_dates.apply(pd.to_datetime)

    # Define the metrics
    metrics = [
        Metric('rmse', root_mean_squared_error, sim_modes),
        Metric('mae', mean_absolute_error, sim_modes),
        Metric('mbe', mean_bias_error, sim_modes),
        Metric('nse', nash_sutcliffe_efficiency, sim_modes),
        Metric('dtw', dynamic_time_warping, sim_modes)
    ]

    # Start a list to store the test station's dataframes
    test_stations_dfs = []

    # Simulate SWE for each station
    for station_idx, (station_name, df_station) in enumerate(dict_dfs.items()):
        print(f"Simulating station {station_idx+1} of {len(cfg.station_names)}.")

        # Take only a fraction of the data if specified
        df_station = df_station.iloc[:int(len(df_station) * (1 - cfg.drop_data))]

        # If in temporal split mode, append the split index to the dataframe
        if cfg.temporal_split and (station_name in cfg.trn_stn):
            df_station['temporal_split'] = 0
            split_dates_stn = split_dates.loc[station_name]
            for i, (start_date, end_date) in enumerate(zip(split_dates_stn['tst_start_date'],
                                                           split_dates_stn['tst_end_date'])):
                tst_cond = (df_station.index >= start_date) & (df_station.index < end_date)
                df_station.loc[tst_cond, 'temporal_split'] = i

        # Initialize a vector for the predicted SWE
        pred_swe_arr = np.zeros((df_station.shape[0], len(dict_models)))

        for mode_idx, (mode, predictors) in enumerate(cfg.modes().items()):
            print(f"Mode {mode_idx + 1} of {len(dict_models)}.")

            # Get the correct model if in station split mode
            model_list = dict_models[mode]
            model = model_list[0]

            # Check if the conditions are met to use a different model
            dif_model = (not cfg.temporal_split) and (mode == 'data_aug') \
                        and (station_name in cfg.tst_stn)
            
            # Get the index of the test station and select the corresponding model
            if dif_model:
                tst_station_idx = cfg.tst_stn.index(station_name)
                model = model_list[tst_station_idx]
            
            # Get the predictor data for the corresponding mode
            df_station_X = df_station.filter(regex=predictors)

            # Get the number of rows in the dataframe
            n_rows = len(df_station_X)

            # Predict each row in the dataframe iteratively
            for row_idx, (df_index, row_serie) in enumerate(df_station_X[:-1].iterrows()):

                # Print progress
                if n_rows >= 10:
                    if row_idx % (n_rows // 5) == 0:
                        progress_pct = row_idx * 100 / n_rows
                        print(f"Progress: {progress_pct:.0f}% completed.")

                # Convert the row to a dataframe
                row = row_serie.to_frame().T

                # Add the previous SWE value to the row
                row['obs_swe'] = pred_swe_arr[row_idx, mode_idx]

                # Decide which model to use in temporal split mode
                if cfg.temporal_split and (station_name in cfg.trn_stn):
                    model = model_list[df_station.loc[df_index, 'temporal_split']]
                
                # Predict the next SWE value and ensure it is non-negative
                pred_dswe = model.predict(row).ravel()
                pred_swe = max(pred_swe_arr[row_idx, mode_idx] + pred_dswe, 0)

                # Save the predicted SWE
                pred_swe_arr[row_idx + 1, mode_idx] = pred_swe
    
        # Save the simulated SWE as a dataframe
        df_swe = pd.DataFrame(pred_swe_arr, index=df_station_X.index, 
                              columns=cfg.modes().keys())
        df_swe.to_csv(paths.temp / f'df_{station_name}_pred_swe.csv')

        # Append the measured and Crocus data to the SWE dataframe
        df_swe['obs_swe'] = df_station['obs_swe']
        df_swe['mod_swe'] = df_station['mod_swe']

        # Create a list with the simulation modes plus Crocus
        sim_modes = ['mod_swe'] + list(cfg.modes().keys())

        # Calculate the nNSE for the training set if in temporal split
        if cfg.temporal_split and (station_name in cfg.trn_stn):

            # Find the start and end dates for the testing set
            tst_start_date = df_split_dates.loc[(station_name, 0), 'tst_start_date']
            tst_end_date = df_split_dates.loc[(station_name, 0), 'tst_end_date']
            split_dates = tst_start_date, tst_end_date

            # Mask the measurements by testing sets
            df_test = mask_measurements_by_year(df_swe.dropna(), 'test', split_dates)
            
            # Calculate and append the metrics
            for metric in metrics:
                metric.calculate_and_append(df_test, station_name)

            # Append df to the list of test stations if it is a train station
            test_stations_dfs.append(df_test)
        
        # Calculate the nNSE for the whole station if in station split
        else:
            for metric in metrics:
                metric.calculate_and_append(df_swe.dropna(), station_name)

            # Append df to the list of test stations if it is a test station
            if (not cfg.temporal_split) and (station_name in cfg.tst_stn):
                test_stations_dfs.append(df_swe)
    
    # Concatenate the DataFrames for the test stations
    df_test = pd.concat(test_stations_dfs)

    # Calculate the metrics for all observations and save the results
    for metric in metrics:
        metric.calculate_and_append(df_test, 'TEST')
        metric.save()

    # Make a folder for the SHAP values
    shap_explanations_path = paths.temp / 'shap_explanations'
    shap_explanations_path.mkdir(exist_ok=True)

    # Loop over each mode and split to calculate the feature importances
    for mode_idx, (mode, predictors) in enumerate(cfg.modes().items()):

        # Retrieve the models, and skip if it is an LSTM
        models = dict_models[mode]
        if models[0].get_model_type() == 'lstm':
            continue

        # Get the training and test data for the SHAP analysis
        trn_dfs = [dict_dfs[station].filter(regex=predictors) for station in cfg.trn_stn]
        tst_dfs = [dict_dfs[station].filter(regex=predictors) for station in cfg.tst_stn]

        # Get the predicted SWE for the SHAP analysis
        trn_swe_dfs = [pd.read_csv(paths.temp / f'df_{station}_pred_swe.csv', index_col=0) for station in cfg.trn_stn]
        tst_swe_dfs = [pd.read_csv(paths.temp / f'df_{station}_pred_swe.csv', index_col=0) for station in cfg.tst_stn]

        # Convert the indices to datetime objects
        trn_swe_dfs = [df.set_index(pd.to_datetime(df.index)) for df in trn_swe_dfs]
        tst_swe_dfs = [df.set_index(pd.to_datetime(df.index)) for df in tst_swe_dfs]

        # Check that the indices are the same for the predictors and SWE dataframes
        assert all([trn_df.index.equals(trn_swe_df.index) for trn_df, trn_swe_df in zip(trn_dfs, trn_swe_dfs)]), \
            "The indices of the training dataframes and the SWE dataframes are not the same."
        assert all([tst_df.index.equals(tst_swe_df.index) for tst_df, tst_swe_df in zip(tst_dfs, tst_swe_dfs)]), \
            "The indices of the test dataframes and the SWE dataframes are not the same."

        # Replace obs_swe column in the training and test dataframes
        for trn_df, trn_swe_df in zip(trn_dfs, trn_swe_dfs):
            trn_df['obs_swe'] = trn_swe_df[mode]
        for tst_df, tst_swe_df in zip(tst_dfs, tst_swe_dfs):
            tst_df['obs_swe'] = tst_swe_df[mode]

        # Initialize the FeatureImportances object
        feature_importances = FeatureImportances(mode, trn_dfs[0].columns)

        for s, model in enumerate(models):
            # Take the corresponding training and test data
            if cfg.temporal_split:
                y = None
                X_trn, X_tst, _, _ = temporal_test_split(trn_dfs, y, s)
            else:
                X_trn, X_tst = pd.concat(trn_dfs), tst_dfs[s]

            # Randomly sample the test data for the explanations, if specified
            X_tst_explain = X_tst.sample(frac=1 - cfg.drop_data_expl)
            if len(X_tst_explain) < 10:
                X_tst_explain = X_tst.sample(n=10)
                
            # Train the SHAP explainer and get the explanation
            print(f"Training the SHAP explainer for mode {mode} and split {s}.")
            explainer = shap.Explainer(model.predict, X_trn)
            try:
                explanation = explainer(X_tst_explain)

            # If the additivity check fails, re-run the explainer without it
            except Exception as e:
                print(e)
                print("Re-running the SHAP explainer with check_additivity=False.")
                explanation = explainer(X_tst, check_additivity=False)

            # Append the explanation to the FeatureImportances object
            feature_importances.append_explanation(explanation, s)
            
        # Save the feature importances to multiple csv files and plot them
        feature_importances.save_to_csv()
        feature_importances.plot_feature_importances()

    return

###############################################################################
# FEATURE IMPORTANCES CLASS
###############################################################################

class FeatureImportances:

    def __init__(self, mode, columns):
        print(f"Initializing feature importances for mode {mode}.")
        self.mode = mode
        self.columns = columns
        self.shap_values = pd.DataFrame(columns=['split_idx'] + list(columns))
        self.base_values = pd.DataFrame(columns=['split_idx', 'base_value'])
        self.data = pd.DataFrame(columns=['split_idx'] + list(columns))

    def append_explanation(self, explanation, split_idx):

        print(f"Appending explanation for split {split_idx}.")
        
        # Convert explanation values to DataFrame and add split_idx column
        explanation_values = pd.DataFrame(explanation.values, columns=self.columns)
        explanation_values['split_idx'] = split_idx
        
        # Convert explanation base values to DataFrame and add split_idx column
        explanation_base_values = pd.DataFrame(explanation.base_values, columns=['base_value'])
        explanation_base_values['split_idx'] = split_idx
        
        # Convert explanation data to DataFrame and add split_idx column
        explanation_data = pd.DataFrame(explanation.data, columns=self.columns)
        explanation_data['split_idx'] = split_idx

        # Concatenate the new explanation values with the existing DataFrames
        self.shap_values = pd.concat([self.shap_values, explanation_values], ignore_index=True)
        self.base_values = pd.concat([self.base_values, explanation_base_values], ignore_index=True)
        self.data = pd.concat([self.data, explanation_data], ignore_index=True)

    def plot_feature_importances(self):

        # Drop the split_idx column from the DataFrames and convert to numpy arrays
        shap_values = self.shap_values.drop(columns='split_idx').values
        base_values = self.base_values.drop(columns='split_idx').values
        data = self.data.drop(columns='split_idx').values

        # Make an explanation object from the stored values
        explanation = shap.Explanation(shap_values, base_values, data,
                                       feature_names=self.columns)

        # Plot the absolute SHAP values as bar plots
        plt.figure(figsize=(16,9))
        shap.plots.bar(explanation, max_display=15, show=False)
        plt.savefig(paths.figures / f'shap_bar_{self.mode}.png', bbox_inches="tight")

        # Plot the SHAP values as violin plots
        plt.figure(figsize=(16,9))
        shap.plots.violin(explanation, max_display=15, show=False)
        plt.savefig(paths.figures / f'shap_vio_{self.mode}.png', bbox_inches="tight")

    def save_to_csv(self):
        print(f"Saving SHAP values to csv files for mode {self.mode}.")
        # Save the DataFrames to csv files
        self.shap_values.to_csv(paths.temp / 'shap_explanations' /
                                f'df_{self.mode}_shap.csv', index=False)
        self.base_values.to_csv(paths.temp / 'shap_explanations'
                                / f'df_{self.mode}_base.csv', index=False)
        self.data.to_csv(paths.temp / 'shap_explanations' /
                         f'df_{self.mode}_data.csv', index=False)
        
###############################################################################
# METRICS CLASS AND FUNCTIONS
###############################################################################

class Metric:
    def __init__(self, name, func, sim_modes):
        self.name = name
        self.func = func
        self.df = pd.DataFrame(columns = sim_modes + ['n'])
        self.sim_modes = sim_modes

    def calculate_and_append(self, data, name):
        self.df.loc[name] = [self.func(data['obs_swe'], data[mode]) for mode in self.sim_modes] + [len(data)]

    def save(self):
        self.df.to_csv(paths.outputs / f'fwd_sim_{self.name}.csv')

###############################################################################

def root_mean_squared_error(obs, sim):
    if len(obs) < 1:
        return float('nan')
    return np.sqrt(np.mean((obs - sim) ** 2))

def mean_absolute_error(obs, sim):
    if len(obs) < 1:
        return float('nan')
    return np.mean(np.abs(obs - sim))

def mean_bias_error(obs, sim):
    if len(obs) < 1:
        return float('nan')
    return np.mean(sim - obs)

def nash_sutcliffe_efficiency(obs, sim):
    if len(obs) < 2:
        return float('nan')
    return 1 - np.sum((obs - sim) ** 2) / np.sum((obs - np.mean(obs)) ** 2)

def dynamic_time_warping(obs, sim):
    if len(obs) < 1:
        return float('nan')
    return fastdtw.dtw(obs, sim, dist=2)[0]/len(obs)