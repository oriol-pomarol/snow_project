import numpy as np
import pandas as pd
from timezonefinder import TimezoneFinder
import ephem
import datetime
import pytz

def data_preprocessing(dfs_obs, dfs_meteo, dfs_model, locations):
    # Define a list to save the ΔSWE measurements
    dfs_obs_delta_swe = []
    # Iterate over stations and variables
    for i, df_obs in enumerate(dfs_obs):
        # Select which variable to use, prioritizing the automatic SWE measurements
        var = df_obs.columns[0]

        # Get the dates where there is available data
        dates = df_obs.index.date

        # Calculate the differences between consecutive dates
        diffs_timedelta = np.diff(dates)
        diff_days = np.array([td.days for td in diffs_timedelta])

        # Find ΔSWE
        delta_swe = df_obs[var][1:].to_numpy() - df_obs[var][:-1].to_numpy()

        # Check if the next measurement is taking at a consecutive day
        consecutive_days = diff_days < 2

        # Check if ΔSWE is not limited by the available SWE
        enough_swe = delta_swe != -1*df_obs[var][:-1].values

        # Subset based on the conditions set above and store as a df
        df_delta_swe = pd.DataFrame({'time': df_obs[np.append(consecutive_days & enough_swe, False)].index, 
                                     'delta_swe':delta_swe[consecutive_days & enough_swe]})
        df_delta_swe.set_index('time', inplace=True)

        # Append delta_swe to list 
        dfs_obs_delta_swe.append(df_delta_swe)

    # Define a list to save the ΔSWE measurements
    dfs_mod_delta_swe = []
    dfs_mod_delta_swe_filt = []

    # Iterate over each DataFrame in dfs_model
    for i, df_mod in enumerate(dfs_model):
        # Find ΔSWE
        delta_swe = df_mod['mod_swe'][1:].to_numpy() - df_mod['mod_swe'][:-1].to_numpy()

        # Check if ΔSWE is not limited by the available SWE
        enough_swe = delta_swe != -1*df_mod['mod_swe'][:-1].values

        # Subset based on the conditions set above and store as a df
        df_delta_swe = pd.DataFrame({'time': df_mod[:-1].index,
                                    'delta_swe':delta_swe})
        df_delta_swe.set_index('time', inplace=True)

        # Append delta_swe to list and print the amount of measurements per station
        dfs_mod_delta_swe.append(df_delta_swe)
        dfs_mod_delta_swe_filt.append(df_delta_swe[enough_swe])


    # Define the names of the aggregated meteorological variables
    names_meteo_agg = ['Psurf_avg', 'Qair_avg', 'Qair_dav', 'Rainf_avg',
                      'Rainf_max', 'Snowf_avg', 'LWdown_int', 'LWdown_dav',
                      'SWdown_int', 'SWdown_dav', 'Tair_avg', 'Tair_int',
                      'Wind_avg', 'Wind_max']

    # Create an empty list for the aggregated meteo DataFrames
    dfs_meteo_agg = []

    for i, df_met in enumerate(dfs_meteo):
        print(f"Station {i+1} of 10.")
        # Find the timezone corresponding to the location of the station
        tf = TimezoneFinder()
        timezone = tf.timezone_at(lat=locations[i,0], lng=locations[i,1])

        # Create an empty dataframe for the aggregated variables
        df_agg = pd.DataFrame()

        # Shift the data 12h to fit the snow obs, and remove the spare data
        df_met.index = df_met.index + pd.Timedelta(hours=12)
        df_met = df_met[11:-13]
        
        for var_name in names_meteo_agg:
            # Take the variable of interest from the original DataFrame
            var = df_met[var_name[:-4]].copy()

            # Aggregate using the indicated operation according to var_name
            if var_name[-3:] == 'avg':
                var_agg = var.resample('D').mean()
            elif var_name[-3:] == 'int':
                var_agg = var.resample('D').apply(positive_integral)
            elif var_name[-3:] == 'max':
                var_agg = var.resample('D').max()
            elif var_name[-3:] == 'dav':
                var_agg = var.resample('D').apply(daytime_average,
                                                  location=locations[i],
                                                  timezone = timezone)
            # Add the variable to the DataFrame
            df_agg[var_name] = var_agg

        # Convert Tair_avg from Kelvin to Celsius
        df_agg['Tair_avg'] = df_agg['Tair_avg'] - 273.15

        # Convert Psurf_avg from Pascal to atmospheres
        df_agg['Psurf_avg'] = df_agg['Psurf_avg'] * 0.00000986923267

        # Convert Qair_avg from kg/kg to g/kg
        df_agg['Qair_avg'] = df_agg['Qair_avg'] * 1000

        # Convert Qair_dav from kg/kg to g/kg
        df_agg['Qair_dav'] = df_agg['Qair_dav'] * 1000

        # Convert Rainf_avg from kg/m²/s to g/m²/s
        df_agg['Rainf_avg'] = df_agg['Rainf_avg'] * 1000

        # Convert Rainf_max from kg/m²/s to g/m²/s
        df_agg['Rainf_max'] = df_agg['Rainf_max'] * 1000

        # Convert Snowf_avg from kg/m²/s to g/m²/s
        df_agg['Snowf_avg'] = df_agg['Snowf_avg'] * 1000

        # Convert SWdown_int from W/m²*h to W/m²*day
        df_agg['SWdown_int'] = df_agg['SWdown_int'] / 24

        # Convert Tair_int from K*h to K*day
        df_agg['Tair_int'] = df_agg['Tair_int'] / 24

        # Convert LWdown_int from W/m²*h to W/m²*day
        df_agg['LWdown_int'] = df_agg['LWdown_int'] / 24

        # Add the DataFrame to the list
        dfs_meteo_agg.append(df_agg)

    return dfs_obs_delta_swe, dfs_meteo_agg, dfs_mod_delta_swe, dfs_mod_delta_swe_filt

####################################################################################
# EXTRA FUNCTIONS
####################################################################################

def calculate_sunrise_sunset(latitude, longitude, date, timezone):
    # Create an observer object
    observer = ephem.Observer()
    observer.lon = longitude
    observer.lat = latitude

    try:
        # Calculate the sunset on the previous day
        previous_day = date - datetime.timedelta(days=1)
        observer.date = previous_day
        previous_sunset = observer.next_setting(ephem.Sun()).datetime()

        # Calculate the sunrise on the given date
        observer.date = date
        sunrise = observer.previous_rising(ephem.Sun()).datetime()

        # Convert times to the specified timezone
        timezone = pytz.timezone(timezone)
        previous_sunset = previous_sunset.astimezone(timezone).time()
        sunrise = sunrise.astimezone(timezone).time()

    except ephem.NeverUpError:
        sunrise = previous_sunset = "SunAlwaysDown"

    except ephem.AlwaysUpError:
        sunrise = previous_sunset = "SunAlwaysUp"

    # Return the previous sunset and the sunrise as time objects in the specified timezone
    return previous_sunset, sunrise

####################################################################################

def is_daytime(sunset, sunrise):
    # Check for the exceptions
    if sunrise == "SunAlwaysDown":
        return np.zeros((24,), dtype=bool)

    if sunrise == "SunAlwaysUp":
        return np.ones((24,), dtype=bool)

    # Extract the hour component from sunrise and sunset times
    sunrise_hour = sunrise.hour
    sunset_hour = sunset.hour

    # Initialize an integer array representing the hours
    hours = np.concatenate((np.arange(12, 24), np.arange(0, 12)))

    # Check if it is daytime for every hour
    if sunset_hour > sunrise_hour:
        daytime = (hours <= sunset_hour) & (hours > sunrise_hour)
    else:
        daytime = (hours <= sunset_hour) | (hours > sunrise_hour)

    # Return a boolean array with the daytime hours
    return daytime

####################################################################################

def daytime_average(array, location, timezone):
    # Compute the sunset and sunrise given the location, date and timezone
    sunset, sunrise = calculate_sunrise_sunset(location[0], location[1],
                                              array.index[0], timezone)
    # Take only the daytime array elements
    daytime_mask = is_daytime(sunset, sunrise)
    daytime_arr = array[daytime_mask]

    # Average the results if there is more than one, otherwise set to 0
    if len(daytime_arr)>0:
        daytime_avg = np.mean(daytime_arr)
    else:
        daytime_avg = 0

    return daytime_avg

####################################################################################

def positive_integral(array):
    # Find the time difference between elements in hours
    time_diff = (array.index[1] - array.index[0]).total_seconds() / 3600

    # Take only the positive array elements
    array_positive = array[array > 0]

    # Find the integral of the positive values
    positive_int = array_positive.sum() * time_diff

    return positive_int