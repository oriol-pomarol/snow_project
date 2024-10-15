import numpy as np
import pandas as pd
import ephem
import datetime
import pytz
from config import cfg, paths

###############################################################################
# GENERAL FUNCTIONS
###############################################################################

def load_processed_data():
    
    # Initialize a dictionary to store the dataframes
    dict_dfs = {}

    # Load the processed data for each station
    for station_name in cfg.station_names:
        filename = f"df_{station_name}_lag_{cfg.lag}.csv"
        df_station = pd.read_csv(paths.proc_data / filename, index_col=0)
        df_station.index = pd.to_datetime(df_station.index)
        dict_dfs[station_name] = df_station
    return dict_dfs

###############################################################################

def preprocess_data_lstm(X):

    # Take the meteorological data
    X_met = X.filter(regex='^met_').values

    # Add extra dimension if the input is 1D
    if X_met.ndim == 1:
        X_met = np.expand_dims(X_met, axis=0)

    # Reshape the array by splitting it along the last axis
    new_shape = X_met.shape[:-1] + (X_met.shape[-1] // cfg.lag, cfg.lag)
    transformed_X_met = X_met.reshape(new_shape)

    # Transpose the subarrays to get the desired structure
    output_X_met = np.transpose(transformed_X_met, axes=(0, 2, 1))

    # If there are additional variables, take them
    X_no_met = X.filter(regex='^(?!met_)').values
    if X_no_met.shape[1] > 0:
        return output_X_met, X_no_met

    return output_X_met


###############################################################################
# METEOROLOGICAL FUNCTIONS
###############################################################################

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

    # Return the previous sunset and the sunrise
    return previous_sunset, sunrise

###############################################################################

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

###############################################################################

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

###############################################################################

def positive_integral(array):
    # Find the time difference between elements in hours
    time_diff = (array.index[1] - array.index[0]).total_seconds() / 3600

    # Take only the positive array elements
    array_positive = array[array > 0]

    # Find the integral of the positive values
    positive_int = array_positive.sum() * time_diff

    return positive_int

###############################################################################

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

###############################################################################

def add_lagged_values(df):
    new_df = df.copy()
    
    for i, col in enumerate(df.columns):
        new_df = pd.concat([new_df.iloc[:, :i*(cfg.lag+1)+1],
                            pd.DataFrame({f'{col}_lag_{j}':df[col].shift(j) \
                                          for j in range(1, cfg.lag+1)}),
                            new_df.iloc[:, i*(cfg.lag+1)+1:]], axis=1)    
    return new_df


###############################################################################
# EXTRA FUNCTIONS
###############################################################################

def find_temporal_split_dates(dfs):
    
    # Define a dataframe to store the split dates
    df_split_dates = pd.DataFrame(columns=['tst_start_date', 'tst_end_date',
                                           'val_start_date', 'val_end_date'])
    
    # Create a MultiIndex for the dataframe
    df_split_dates.index = pd.MultiIndex.from_tuples([], names=['station', 'split'])

    # Retrieve the number of splits
    n_splits = cfg.n_temporal_splits

    for station_idx, df in enumerate(dfs):

        # Shift the index by 6 months to start the year in July
        shifted_index = df.index - pd.offsets.DateOffset(months=6)

        # Get a list of years with more than 20% of available data
        years = shifted_index.year.unique()
        years = [year for year in years if sum(shifted_index.year == year) > 0.2 * 365]

        # Get the timestamps starting in July
        timestamps = [pd.Timestamp(f'{year}-07-01') for year in years]

        # Add the last year plus one to the list
        timestamps.append(timestamps[-1] + pd.DateOffset(years=1))

        # Calculate the number of validation years
        val_years = max(1, int(cfg.val_ratio * len(timestamps) * (n_splits - 1) / n_splits))

        # Iterate over the split dates
        for split_idx in range(n_splits):

            # Take the start test dates from the cv splits
            tst_start_date = timestamps[split_idx * len(timestamps) // n_splits]

            # Take the end test date and validation dates depending on the split index
            if split_idx == n_splits - 1:
                tst_end_date = timestamps[-1]
                val_end_date = tst_start_date
            else:
                val_end_date = timestamps[-1]
                tst_end_date = timestamps[(split_idx + 1) * len(timestamps) // n_splits]
            
            # Calculate the start validation date
            val_start_date = timestamps[timestamps.index(val_end_date) - val_years]
            
            # Save the split dates
            df_split_dates.loc[(cfg.trn_stn[station_idx], split_idx), :] = \
                [tst_start_date, tst_end_date, val_start_date, val_end_date]

    # Save the train_test split dates as a csv
    df_split_dates.to_csv(paths.temp_data / 'split_dates.csv')

    return

###############################################################################

def data_aug_split(X_trn, y_trn, X_aug, y_aug):

    # Concatenate the augmented data
    X_aug = pd.concat(X_aug)
    y_aug = pd.concat(y_aug)

    # Change the name of the augmented data to the target name
    y_aug = y_aug.rename(columns={y_aug.columns[0] : y_trn.columns[0]})
    X_aug = X_aug.rename(columns={X_aug.columns[-1] : X_trn.columns[-1]})
    
    # Calculate the training weights of the modelled data
    weight_aug = cfg.rel_weight * len(y_trn) / len(y_aug)
    sample_weight = np.concatenate((np.ones(len(y_trn)), 
                                    np.full(len(y_aug), weight_aug)))
    
    # Concatenate the observed and augmented datasets
    X_trn = pd.concat([X_trn, X_aug])
    y_trn = pd.concat([y_trn, y_aug])

    return X_trn, y_trn, sample_weight