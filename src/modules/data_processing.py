import xarray as xr
import os
import pandas as pd
from timezonefinder import TimezoneFinder
from os import listdir
from config import cfg, paths
from .auxiliary_functions import (
    positive_integral,
    daytime_average,
    change_meteo_units,
    add_lagged_values,
)

def data_processing():

    # Read the station data
    df_stations = pd.read_csv(
        paths.raw_data / "Menard_Essery_2019.tab",
        delimiter="\t",
        skiprows=35
    )

    # Save the in-situ meteo and observed data as separate dataframes
    data_info_met = df_stations[10:20].reset_index()
    data_info_obs = df_stations[20:30].reset_index()

    # Iterate over the stations
    for station_idx, station_name in enumerate(cfg.station_names):
        print(
            "Loading and preprocessing data from station "
            f"{station_idx + 1} of {len(cfg.station_names)}..."
        )

        # Get the paths to the files (meteo and observed data)
        filename_met = data_info_met["File name"][station_idx]
        filename_obs = data_info_obs["File name"][station_idx]

        file_path_met = paths.raw_data / "ESM-SnowMIP_all" / f"{filename_met}.nc"
        file_path_obs = paths.raw_data / "ESM-SnowMIP_all" / f"{filename_obs}.nc"

        # Obtain a pandas DataFrame (meteo and observed data)
        df_met = xr.open_dataset(file_path_met).to_dataframe()
        df_obs = xr.open_dataset(file_path_obs).to_dataframe()

        # Obtain the dataset (model data)
        dir_path_mod = paths.raw_data / "simus_CROCUS" / station_name
        file_path_mod = os.path.join(dir_path_mod, listdir(dir_path_mod)[0])
        dataset_mod = xr.open_dataset(file_path_mod, decode_times=False)

        # Get the location of the station
        lat_station = data_info_met.loc[station_idx, "Latitude"]
        lng_station = data_info_met.loc[station_idx, "Longitude"]

        # Pre-process the data
        df_met_preprocessed = met_preprocessing(df_met, lat_station, lng_station)
        df_obs_preprocessed = obs_preprocessing(df_obs)
        df_mod_preprocessed = mod_preprocessing(dataset_mod)

        # Concatenate the DataFrames
        df_data = pd.concat(
            [df_met_preprocessed, df_obs_preprocessed, df_mod_preprocessed],
            axis=1
        )

        # Create new column representing delta SWE (observed and model data)
        df_data["delta_obs_swe"] = df_data["obs_swe"].diff().shift(-1)
        df_data["delta_mod_swe"] = df_data["mod_swe"].diff().shift(-1)

        # Save the DataFrame
        df_data.to_csv(paths.proc_data / f"df_{station_name}_lag_{cfg.lag}.csv")

    return


###############################################################################
# DATA PROCESSING FUNCTIONS
###############################################################################

def obs_preprocessing(df_obs):
    # Take the best available SWE measurements at each station and rename them
    if "snw_auto" in df_obs.columns:
        df_obs = df_obs[["snw_auto"]].rename(columns={"snw_auto": "obs_swe"})
    else:
        df_obs = df_obs[["snw_man"]].rename(columns={"snw_man": "obs_swe"})

    # Remove unnecessary measurements from the same time value
    df_obs = df_obs.groupby("time").first()

    # Take the measurements at 12:00
    df_obs = df_obs[df_obs.index.strftime("%H:%M") == "12:00"]

    # Drop the hour from the index
    df_obs.index = pd.to_datetime(df_obs.index.date)

    return df_obs


###############################################################################
def mod_preprocessing(dataset_mod):
    # Find the starting datee for each simulation
    time_var = dataset_mod.variables.get("time")
    start_date = pd.to_datetime(time_var.attrs.get("units")[12:])

    # Select the total SWE only
    dataset_mod_swe = dataset_mod["WSN_T_ISBA"]

    # Convert to a DataFrame and rename the total SWE
    df_mod = dataset_mod_swe.to_dataframe()
    df_mod.rename(columns={"WSN_T_ISBA": "mod_swe"}, inplace=True)

    # Take only the first sample for each unique time value
    df_mod = df_mod.groupby("time").first()

    # Convert the time index to datetime values starting at start_date
    start_datetime = pd.to_datetime(start_date)
    datetime_index = pd.date_range(start=start_datetime,
                                   periods=len(df_mod),
                                   freq="H")
    df_mod.index = datetime_index

    # Take only the measurements made at 12:00
    df_mod = df_mod[df_mod.index.strftime("%H:%M") == "12:00"]

    # Drop the hour from the index
    df_mod.index = pd.to_datetime(df_mod.index.date)

    return df_mod


###############################################################################


def met_preprocessing(df_met, lat_station, lng_station):
    # Define the names of the aggregated meteorological variables
    names_met_agg = [
        "Psurf_avg",
        "Qair_avg",
        "Qair_dav",
        "Rainf_avg",
        "Rainf_max",
        "Snowf_avg",
        "LWdown_int",
        "LWdown_dav",
        "SWdown_int",
        "SWdown_dav",
        "Tair_avg",
        "Tair_int",
        "Wind_avg",
        "Wind_max",
    ]

    # Find the timezone corresponding to the location of the station
    tf = TimezoneFinder()
    timezone = tf.timezone_at(lat=lat_station, lng=lng_station)

    # Create an empty dataframe for the aggregated variables
    df_agg = pd.DataFrame()

    # Shift the data 12h to fit snow observations
    df_met.index = df_met.index - pd.Timedelta(hours=12)

    # Remove rows from incomplete days
    while df_met.index[0].hour != 0:
        df_met = df_met[1:]
    while df_met.index[-1].hour != 23:
        df_met = df_met[:-1]

    for var_name in names_met_agg:
        # Take the variable of interest from the original DataFrame
        var = df_met[var_name[:-4]].copy()

        # Aggregate using the indicated operation according to var_name
        if var_name[-3:] == "avg":
            var_agg = var.resample("D").mean()
        elif var_name[-3:] == "int":
            var_agg = var.resample("D").apply(positive_integral)
        elif var_name[-3:] == "max":
            var_agg = var.resample("D").max()
        elif var_name[-3:] == "dav":
            var_agg = var.resample("D").apply(
                daytime_average,
                lat_station=lat_station,
                lng_station=lng_station,
                timezone=timezone,
            )
        # Add the variable to the DataFrame
        df_agg['met_' + var_name] = var_agg

    # Change the units of the aggregated variables
    df_agg = change_meteo_units(df_agg)

    # Add lagged values to the DataFrame
    df_agg_lagged = add_lagged_values(df_agg)

    return df_agg_lagged