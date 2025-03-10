import xarray as xr
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
        file_path_mod = dir_path_mod / listdir(dir_path_mod)[0]
        dataset_mod = xr.open_dataset(file_path_mod, decode_times=False)

        # Convert the time variable to a datetime object
        time_var = dataset_mod.variables.get("time")
        start_date = pd.to_datetime(time_var.attrs.get("units")[12:])
        time = pd.date_range(start=start_date, periods=len(time_var), freq="H")
        dataset_mod = dataset_mod.assign_coords(time=time)

        # Get the location of the station
        lat_station = data_info_met.loc[station_idx, "Latitude"]
        lng_station = data_info_met.loc[station_idx, "Longitude"]

        # Pre-process the data
        df_met_preprocessed = met_preprocessing(df_met, lat_station, lng_station)
        df_obs_preprocessed = obs_preprocessing(df_obs)
        df_mod_preprocessed = mod_preprocessing(dataset_mod)
        df_cro_preprocessed = cro_preprocessing(dataset_mod)

        # Concatenate the DataFrames
        df_data = pd.concat(
            [df_met_preprocessed, df_cro_preprocessed,
             df_obs_preprocessed, df_mod_preprocessed],
            axis=1
        )

        # Create new column representing delta SWE (observed and modelled)
        df_data["delta_obs_swe"] = df_data["obs_swe"].diff().shift(-1)
        df_data["delta_mod_swe"] = df_data["mod_swe"].diff().shift(-1)

        # Save the DataFrame
        df_data.to_csv(paths.proc_data / f"df_{station_name}_lag_{cfg.lag}.csv")

    return


###############################################################################
# DATA PROCESSING FUNCTIONS
###############################################################################

def obs_preprocessing(df_obs):
    """
    Return the automatic SWE measurements at 12:00 or manual if not available.

    Parameters:
    df_obs (pd.DataFrame): DataFrame containing the observed SWE data.

    Returns:
    df_obs (pd.DataFrame): DataFrame containing the processed observed SWE data.
    """
    # Take automatic SWE measurements if available and rename them
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
    """
    Return the Crocus SWE measurements at 12:00.

    Parameters:
    dataset_mod (xr.Dataset): Dataset containing the Crocus data.

    Returns:
    df_mod (pd.DataFrame): DataFrame containing the processed Crocus SWE data.
    """

    # Convert total SWE to a DataFrame and rename to mod_swe
    df_mod = dataset_mod["WSN_T_ISBA"].to_dataframe()
    df_mod.rename(columns={"WSN_T_ISBA": "mod_swe"}, inplace=True)

    # Take only the first sample for each unique time value
    df_mod = df_mod.groupby("time").first()

    # Take only the measurements made at 12:00
    df_mod = df_mod[df_mod.index.strftime("%H:%M") == "12:00"]

    # Drop the hour from the index
    df_mod.index = pd.to_datetime(df_mod.index.date)

    return df_mod

###############################################################################

def cro_preprocessing(dataset_mod):
    """
    Return the daily aggregated Crocus snowpack state variables.

    Parameters:
    dataset_mod (xr.Dataset): Dataset containing the Crocus data.

    Returns:
    df_agg (pd.DataFrame): DataFrame containing the aggregated Crocus data.
    """
    # Define the names of the aggregated meteorological variables
    names_cro_agg = [
        "TG1_avg",
        "WG1_avg",
        "WGI1_avg",
        "RN_ISBA_avg",
        "LE_ISBA_avg",
        "LEI_ISBA_avg",
        "SWD_ISBA_avg",
        "TS_ISBA_avg",
        "TS_ISBA_max",
        "RAM_SONDE_avg",
        "WET_TH_avg",
        "REFROZ_TH_avg",
        "PSN_ISBA_avg",
        "TALB_ISBA_avg",
        "DSN_T_ISBA_pts",
    ]

    # Create an empty dataframe for the aggregated variables
    df_agg = pd.DataFrame()

    # Take the variables of interest from the original dataset
    unique_vars = list(set([var[:-4] for var in names_cro_agg]))
    df_cro = dataset_mod[unique_vars].to_dataframe()

    # Take only the first sample for each unique time value
    df_cro = df_cro.groupby(level="time").first()

    # Shift the data 12h to fit snow observations
    df_cro.index = df_cro.index - pd.Timedelta(hours=12)

    # Remove rows from incomplete days
    while df_cro.index[0].hour != 0:
        df_cro = df_cro[1:]
    while df_cro.index[-1].hour != 23:
        df_cro = df_cro[:-1]

    for var_name in names_cro_agg:
        # Take the variable of interest from the original DataFrame
        var = df_cro[var_name[:-4]].copy()

        # Aggregate using the indicated operation according to var_name
        if var_name[-3:] == "avg":
            var_agg = var.resample("D").mean()
        elif var_name[-3:] == "max":
            var_agg = var.resample("D").max()
        elif var_name[-3:] == "pts":
            var_agg = var.resample("D").first()

        # Add the variable to the DataFrame
        df_agg[var_name] = var_agg

    # Retrieve the layer-based variables from the original dataset
    df_lbv = dataset_mod[["WSN_VEG", "SNOWTEMP", "SNOWLIQ", "SNOWDZ"]].to_dataframe()

    # Calculate the snow saturation
    liquid_water_content = df_lbv["SNOWLIQ"].groupby(level="time").sum()
    snow_depth = df_lbv["SNOWDZ"].groupby(level="time").sum()
    saturation = liquid_water_content / snow_depth

    # Calculate the cold content
    temp_diff = df_lbv["SNOWTEMP"] - 273.15 # K
    ice_heat_cpt = 2100 # J/kg/K
    cold_content_lyr = ice_heat_cpt * df_lbv["WSN_VEG"] * temp_diff
    cold_content = cold_content_lyr.groupby(level="time").sum()

    for var in [saturation, cold_content]:
        
        # Shift the data 12h to fit snow observations
        var.index = var.index - pd.Timedelta(hours=12)

        # Remove rows from incomplete days
        while var.index[0].hour != 0:
            var = var[1:]
        while var.index[-1].hour != 23:
            var = var[:-1]

    # Aggregate the layer-based variables to daily values
    df_agg["SNOW_SAT_avg"] = saturation.resample("D").mean()
    df_agg["COLD_CONTENT_pts"] = cold_content.resample("D").first()

    # Fill the missing values in the snow saturation with zeros
    df_agg["SNOW_SAT_avg"].fillna(0, inplace=True)

    # Add cro_ as prefix to the variable names
    df_agg.columns = [f"cro_{col}" for col in df_agg.columns]

    return df_agg

###############################################################################

def met_preprocessing(df_met, lat_station, lng_station):
    """
    Return the daily aggregated meteorological variables.

    Parameters:
    df_met (pd.DataFrame): DataFrame containing the meteorological data.
    lat_station (float): Latitude of the station.
    lng_station (float): Longitude of the station.

    Returns:
    df_agg_lagged (pd.DataFrame): DataFrame containing the processed
    meteorological data.
    """
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

        # If the variable is the temperature, convert it to Celsius
        if var_name[:4] == "Tair":
            var -= 273.15

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
        df_agg[var_name] = var_agg

    # Change the units of the aggregated variables
    df_agg = change_meteo_units(df_agg)

    # Add met_ as prefix to the variable names
    df_agg.columns = [f"met_{col}" for col in df_agg.columns]

    # Add lagged values to the DataFrame
    df_agg_lagged = add_lagged_values(df_agg)

    return df_agg_lagged