import numpy as np
import pandas as pd
import ephem
import datetime
import pytz
from config import cfg, paths

###############################################################################
# METEOROLOGICAL FUNCTIONS
###############################################################################

def calculate_sunrise_sunset(latitude, longitude, date, timezone):
    """
    Calculate the sunrise and sunset times for a given location and date.
    
    Parameters:
    latitude (float): The latitude of the location in degrees.
    longitude (float): The longitude of the location in degrees.
    date (datetime): The date for which to calculate the sunrise and sunset.
    timezone (str): The timezone of the location.

    Returns:
    previous_sunset (datetime.time): The time of the previous sunset.
    sunrise (datetime.time): The time of the sunrise.
    """
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
    """
    Check if the current hour is during the day.

    Parameters:
    sunset (datetime.time): The time of the sunset.
    sunrise (datetime.time): The time of the sunrise.

    Returns:
    daytime (np.array): A boolean array indicating if the current hour is during the day.
    """
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
    """
    Calculate the average of the array elements during the daytime.

    Parameters:
    array (pd.Series): The array of values to average.
    lat_station (float): The latitude of the station in degrees.
    lng_station (float): The longitude of the station in degrees.
    timezone (str): The timezone of the station.

    Returns:
    daytime_avg (float): The average of the array elements during the daytime.
    """
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
    """
    Calculate the integral in time of the positive values in the array.

    Parameters:
    array (pd.Series): The array of values to integrate.

    Returns:
    positive_int (float): The integral of the positive values in the array.
    """
    # Find the time difference between elements in hours
    time_diff = (array.index[1] - array.index[0]).total_seconds() / 3600

    # Take only the positive array elements
    array_positive = array[array > 0]

    # Find the integral of the positive values
    positive_int = array_positive.sum() * time_diff

    return positive_int

###############################################################################

def change_meteo_units(df_agg):
    """
    Convert the meteorological variables to the desired units.

    Parameters:
    df_agg (pd.DataFrame): The aggregated dataframe with the meteorological variables.

    Returns:
    df_agg (pd.DataFrame): The aggregated dataframe with the converted meteorological variables.
    """

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

    # Convert LWdown_int from W/m²*h to W/m²*day
    df_agg['LWdown_int'] = df_agg['LWdown_int'] / 24

    return df_agg

###############################################################################

def add_lagged_values(df):
    """
    Add columns containing the lagged values of the dataframe variables.

    Parameters:
    df (pd.DataFrame): The dataframe with the meteorological variables.

    Returns:
    new_df (pd.DataFrame): The dataframe with the lagged values.
    """
    # Create a copy of the dataframe
    new_df = df.copy()
    
    # Iterate over the columns and add the lagged values
    for i, col in enumerate(df.columns):
        new_df = pd.concat([new_df.iloc[:, :i*(cfg.lag+1)+1],
                            pd.DataFrame({f'{col}_lag_{j}':df[col].shift(j) \
                                          for j in range(1, cfg.lag+1)}),
                            new_df.iloc[:, i*(cfg.lag+1)+1:]], axis=1)    
    return new_df

###############################################################################
# DATA PROCESSING FUNCTIONS
###############################################################################

def load_processed_data():
    """
    Load the processed data for each station.

    Returns:
    dict_dfs (dict): A dictionary containing the dataframes for each station.
    """
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
    """
    Preprocess the input data for the LSTM model.

    Parameters:
    X (pd.DataFrame): The input dataframe with the meteorological variables.

    Returns:
    output_X_met (np.array): The preprocessed input array with the meteorological variables.
    """
    # Split between meteorological and additional variables
    X_met = X.filter(regex='^met_').values
    X_add = X.filter(regex='^(?!met_)').values

    # Add extra dimension if the input is 1D
    if X_met.ndim == 1:
        X_met = np.expand_dims(X_met, axis=0)

    # Reshape the array by splitting it along the last axis
    new_shape = X_met.shape[:-1] + (X_met.shape[-1] // cfg.lag, cfg.lag)
    transformed_X_met = X_met.reshape(new_shape)

    # Transpose the subarrays to get the desired structure
    output_X_met = np.transpose(transformed_X_met, axes=(0, 2, 1))

    # If available, return the additonal variables as well 
    if X_add.shape[1] > 0:
        return output_X_met, X_add

    return output_X_met

###############################################################################

def integrate_aug_data(X_trn, y_trn, X_aug, y_aug, rel_weight):
    """
    Integrate the augmented data into the training dataset.

    Parameters:
    X_trn (pd.DataFrame): The training dataframe with the meteorological variables.
    y_trn (pd.Series): The training series with the observed SWE values.
    X_aug (list): The list of dataframes with the augmented meteorological variables.
    y_aug (list): The list of series with the augmented observed SWE values.
    rel_weight (float): The relative weight of the augmented data.

    Returns:
    X_trn (pd.DataFrame): The training dataframe with the integrated augmented meteorological variables.
    y_trn (pd.DataFrame): The training series with the integrated augmented observed SWE values.
    sample_weight (np.array): The training weights of the data.
    """
    # Concatenate the augmented data
    X_aug = pd.concat(X_aug)
    y_aug = pd.concat(y_aug)
    
    # Calculate the training weights of the modelled data
    weight_aug = rel_weight * len(y_trn) / len(y_aug)
    sample_weight = np.concatenate((np.ones(len(y_trn)), 
                                    np.full(len(y_aug), weight_aug)))
    
    # Concatenate the observed and augmented datasets
    X_trn = pd.concat([X_trn, X_aug])
    y_trn = pd.concat([y_trn, y_aug])

    return X_trn, y_trn, sample_weight

###############################################################################

def replace_obs_dropna(aug_df):
    """
    Replace the observed SWE values with the modelled ones and drop the rows with NAs.
    
    Parameters:
    aug_df (pd.DataFrame): The augmented dataframe with the observed and modelled SWE values.

    Returns:
    aug_df (pd.DataFrame): The clean augmented dataframe with the replaced SWE values.
    """
    # Make a copy of the dataframe
    aug_df = aug_df.copy()
    
    # Drop the observed SWE and derived columns
    aug_df.drop(columns=['obs_swe', 'delta_obs_swe'], inplace=True)

    # Rename the modeled SWE and derived columns
    aug_df.rename(columns={'mod_swe': 'obs_swe',
                        'delta_mod_swe': 'delta_obs_swe'}, inplace=True)
    
    # Drop the rows with NAs
    aug_df.dropna(inplace=True)

    return aug_df

###############################################################################
# DATA SPLIT FUNCTIONS
###############################################################################

def find_temporal_split_dates(dfs):
    """
    Find the temporal split dates for the training and validation data and save them as a csv.

    Parameters:
    dfs (list): A list containing the dataframes for each station.
    """
    # Create an empty list to contain each station's dates
    station_dates = []

    # Retrieve the number of splits
    n_splits = cfg.n_temporal_splits

    for station_idx, df in enumerate(dfs):

        # Define a dataframe to store the split dates
        df_split_dates = pd.DataFrame(columns=['tst_start_date', 'tst_end_date'])

        # Create a MultiIndex for the dataframe
        df_split_dates.index = pd.MultiIndex.from_tuples([], names=['station', 'split'])

        # Shift the index by 6 months to start the year in July
        shifted_index = df.index - pd.offsets.DateOffset(months=6)

        # Get a list of years with more than 20% of available data
        years = shifted_index.year.unique()
        years = [year for year in years if sum(shifted_index.year == year) > 0.2 * 365]

        # Get the timestamps starting in July
        timestamps = [pd.Timestamp(f'{year}-07-01') for year in years]

        # Add the last year plus one to the list
        timestamps.append(timestamps[-1] + pd.DateOffset(years=1))

        # Iterate over the split dates
        for split_idx in range(n_splits):

            # Take the start test dates from the cv splits
            tst_start_date = timestamps[split_idx * len(timestamps) // n_splits]

            # Take the end test date and validation dates depending on the split index
            if split_idx == n_splits - 1:
                tst_end_date = timestamps[-1]
            else:
                tst_end_date = timestamps[(split_idx + 1) * len(timestamps) // n_splits]
            
            # Save the split dates
            df_split_dates.loc[(cfg.trn_stn[station_idx], split_idx), :] = \
                [tst_start_date, tst_end_date]
            
        # Set the first starting date as 01-01-1900 and the last ending date as 31-12-2100
        df_split_dates.loc[(cfg.trn_stn[station_idx], 0), 'tst_start_date'] = pd.Timestamp('1900-01-01')
        df_split_dates.loc[(cfg.trn_stn[station_idx], n_splits-1), 'tst_end_date'] = pd.Timestamp('2100-12-31')

        # Add the validation dates by shifting the test dates one split
        df_split_dates['val_start_date'] = np.roll(df_split_dates['tst_start_date'], -1)
        df_split_dates['val_end_date'] =  np.roll(df_split_dates['tst_end_date'], -1)

        # Append the dataframe to the list
        station_dates.append(df_split_dates)

    # Concatenate the split dates for all stations
    df_split_dates = pd.concat(station_dates)

    # Save the train_test split dates as a csv
    df_split_dates.to_csv(paths.temp / 'split_dates.csv')

    return

###############################################################################

def station_validation_split(X, y, split_idx):
    """
    Split the data into training and validation sets for a station split.
    
    Parameters:
    X (list): A list containing the predictor variable dataframes for each station.
    y (list): A list containing the target variable dataframes for each station.
    split_idx (int): The index of the cross validation split.

    Returns:
    X_trn (pd.DataFrame): The training dataframe with the predictor variables.
    X_val (pd.DataFrame): The validation dataframe with the predictor variables.
    y_trn (pd.Series): The training series with the target variables.
    y_val (pd.Series): The validation series with the target variables.
    """
    # Take one station for testing
    X_tst = X[split_idx]
    y_tst = y[split_idx]

    # Concatenate the remaining stations for training
    X_trn = pd.concat([X[j] for j in range(len(X)) if j!=split_idx])
    y_trn = pd.concat([y[j] for j in range(len(y)) if j!=split_idx])

    return X_trn, X_tst, y_trn, y_tst

###############################################################################

def temporal_validation_split(X, y, split_idx):
    """
    Split the data into training and validation sets for a temporal split.
    
    Parameters:
    X (list): A list containing the predictor variable dataframes for each station.
    y (list): A list containing the target variable dataframes for each station.
    split_idx (int): The index of the cross validation split.

    Returns:
    X_trn (pd.DataFrame): The training dataframe with the predictor variables.
    X_val (pd.DataFrame): The validation dataframe with the predictor variables.
    y_trn (pd.Series): The training series with the target variables.
    y_val (pd.Series): The validation series with the target variables.
    """
    # Specify the columns that should be parsed as dates
    date_columns = ['tst_start_date', 'tst_end_date', 'val_start_date', 'val_end_date']

    # Load the split dates
    df_split_dates = pd.read_csv(paths.temp / 'split_dates.csv', index_col=[0, 1], parse_dates=date_columns)

    # Initialize lists to store the training and validation data
    X_trn, y_trn, X_val, y_val = [], [], [], []

    for i, station in enumerate(cfg.trn_stn):

        # Retrieve the split dates for the current station and split
        tst_start_date, tst_end_date, val_start_date, val_end_date = \
            df_split_dates.loc[(station, split_idx)].values
        
        # Get the trn/val conditions for the current station and split
        trn_cond = ((X[i].index < tst_start_date) | \
                    (X[i].index >= tst_end_date)) & \
                   ((X[i].index < val_start_date) | \
                    (X[i].index >= val_end_date))
     
        val_cond = (X[i].index >= val_start_date) & \
                   (X[i].index < val_end_date)

        # Append the training and validation data
        X_trn.append(X[i].loc[trn_cond])
        y_trn.append(y[i].loc[trn_cond])
        X_val.append(X[i].loc[val_cond])
        y_val.append(y[i].loc[val_cond])        

    # Concatenate the training and validation data
    X_trn, y_trn = pd.concat(X_trn), pd.concat(y_trn)
    X_val, y_val = pd.concat(X_val), pd.concat(y_val)

    return X_trn, X_val, y_trn, y_val

###############################################################################

def temporal_test_split(X, y, split_idx):
    """
    Split the data into training and test sets for a temporal split.

    Parameters:
    X (list): A list containing the predictor variable dataframes for each station.
    y (list): A list containing the target variable dataframes for each station.
    split_idx (int): The index of the cross validation split.

    Returns:
    X_trn (pd.DataFrame): The training dataframe with the predictor variables.
    X_tst (pd.DataFrame): The test dataframe with the predictor variables.
    y_trn (pd.Series): The training series with the target variables.
    y_tst (pd.Series): The test series with the target variables.
    """
    # Specify the columns that should be parsed as dates
    date_columns = ['tst_start_date', 'tst_end_date', 'val_start_date', 'val_end_date']

    # Load the split dates
    df_split_dates = pd.read_csv(paths.temp / 'split_dates.csv',
                                 index_col=[0, 1], parse_dates=date_columns)

    # Initialize lists to store the training and validation data
    X_trn, y_trn, X_tst, y_tst = [], [], [], []

    for i, station in enumerate(cfg.trn_stn):

        # Retrieve the split dates for the current station and split
        tst_start_date, tst_end_date, _, _ = \
            df_split_dates.loc[(station, split_idx)].values
        
        # Filter the trn and tst data conditions for the current station and split
        trn_cond = (X[i].index < tst_start_date) | \
                   (X[i].index >= tst_end_date)
        tst_cond = (X[i].index >= tst_start_date) & \
                   (X[i].index < tst_end_date)
        # Append the training and test data
        X_trn.append(X[i].loc[trn_cond])
        X_tst.append(X[i].loc[tst_cond])
        if y is not None:
            y_trn.append(y[i].loc[trn_cond])
            y_tst.append(y[i].loc[tst_cond])

    # Concatenate the training and test data
    X_trn, X_tst = pd.concat(X_trn), pd.concat(X_tst)
    if y is not None:
        y_trn, y_tst = pd.concat(y_trn), pd.concat(y_tst)

    return X_trn, X_tst, y_trn, y_tst

###############################################################################
# ADDITIONAL FUNCTIONS
###############################################################################

def mask_measurements_by_year(df, year, split_dates=None):
    """
    Mask the dataframe by year or train/test split.

    Parameters:
    df (pd.DataFrame): The dataframe to mask.
    year (str): The year to mask the dataframe.
    split_dates (list): The list of split dates for the training and test data.

    Returns:
    df (pd.DataFrame): The masked dataframe.
    """
    # If the dataframe is empty or the year is 'all', return the dataframe
    if (len(df) == 0) or (year == 'all'):
        return df

    # If the year is 'train', 'test', or a specific year, mask the data
    elif year == 'train':
        mask = (df.index < split_dates[0]) | (df.index >= split_dates[1])

    elif year == 'test':
        mask = (df.index >= split_dates[0]) & (df.index < split_dates[1])

    elif year.isdigit():
        year = int(year)
        start_date = pd.to_datetime(f'{year}-07-01')
        end_date = pd.to_datetime(f'{year + 1}-07-01')
        mask = (df.index >= start_date) & (df.index < end_date)
        
    else:
        raise ValueError(f'Invalid input year: {year}')
    
    return df[mask]

###############################################################################

def drop_samples(dfs, drop_pct, min_samples=10):
    """
    Drop a percentage of samples from the dataframes.

    Parameters:
    dfs (list): A list containing the dataframes to drop samples from.
    drop_pct (float): The percentage of samples to drop.
    min_samples (int): The minimum number of samples to keep.

    Returns:
    dfs (list): A list containing the dataframes with the dropped samples.
    """
    
    # Set the number of samples to keep, ensuring it is at least min_samples
    n_end_samples = max(min_samples, int(len(dfs[0]) * (1 - drop_pct)))

    # Create a mask to keep a random subset of the data with n_end_samples
    mask = np.zeros(len(dfs[0]), dtype=bool)
    mask[-n_end_samples:] = True
    np.random.shuffle(mask)

    # Apply the mask to the dataframes, skip if it is None
    dfs = [df[mask] if df is not None else None for df in dfs]

    return dfs

###############################################################################

def get_cv_info(mode):
    """
    Report the number of cross validation splits and suffix.

    Parameters:
    mode (str): The simulation mode (e.g., 'dir_pred', 'post_prc').

    Returns:
    n_splits (int): The number of cross validation splits.
    suffix (str): The suffix to append to the output files.
    """
    # Default is 1 split and no suffix
    n_splits = 1
    suffix = ''

    # If in temporal mode, use the value from config and set the suffix
    if cfg.temporal_split:
        n_splits = cfg.n_temporal_splits
        suffix = 'temp_split'

    # If in data augmentation mode, use the number of test stations
    elif mode == 'data_aug':
        n_splits = len(cfg.tst_stn)
        suffix = 'aug_split'

    return n_splits, suffix