import numpy as np
import pandas as pd
import ephem
import datetime
import pytz

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

def daytime_average(array, lat_station, lng_station, timezone):
    # Compute the sunset and sunrise given the location, date and timezone
    sunset, sunrise = calculate_sunrise_sunset(lat_station, lng_station,
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

####################################################################################

def change_meteo_units(df_agg):

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

    return df_agg

####################################################################################

def add_lagged_values(df, lag):
    new_df = df.copy()
    
    for i, col in enumerate(df.columns):
        new_df = pd.concat([new_df.iloc[:, :i*(lag+1)+1],
                            pd.DataFrame({f'{col}_lag_{j}':df[col].shift(j) \
                                          for j in range(1, lag+1)}),
                            new_df.iloc[:, i*(lag+1)+1:]], axis=1)    
    return new_df