"""
AIS Data Preprocessing Utilities

Author: Md Mahbub Alam

Description:
    Reusable functions for AIS dataset creation, noise filtering,
    time-based trip segmentation, Cubic Hermite interpolation, and
    kinematic feature engineering.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, PchipInterpolator


# Columns used in preprocessing and interpolation
RAW_COLUMNS = [
    'mmsi', 'timestamp', 'lat', 'lon', 'sog', 'cog', 'rot', 'ship_type'
]
COLS = ['mmsi', 'timestamp', 'lat', 'lon', 'sog', 'cog', 'ship_type']
VESSEL_TYPES = [
    'cargo', 'fishing', 'tanker', 'tug', 'passenger', 'pcraft', 'sailing',
    'towing'
]

#Dictionary of vessel types with codes
DICT_VESSEL_TYPE = {
    'fishing': [30],
    'towing': [31, 32],
    'sailing': [36],
    'pcraft': [37],
    'tug': [52],
    'passenger': [60, 61, 62, 63, 64, 65, 66, 67, 68, 69],
    'cargo': [70, 71, 72, 73, 74, 75, 76, 77, 78, 79],
    'tanker': [80, 81, 82, 83, 84, 85, 86, 87, 88, 89]
}


# -----------------------------------------------------------------------------
# Dataset creation and initial preprocessing

# Return corresponding vessel type string for a vessel code
def get_vessel_type(code):
    """Return the corresponding vessel type string for an AIS code."""
    for k, v in DICT_VESSEL_TYPE.items():
        if code in v:
            return k
    return 'other'


# Replace vessel code with vessel type
def replace_code_with_vessel_type(ais_df):
    """Replace numeric ship-type codes with their vessel-type names."""
    for index, row in ais_df.iterrows():
        ais_df.at[index, 'ship_type'] = get_vessel_type(
            int(float(row.ship_type))
        )
    return ais_df


# filter out trajectories where ship_type is missing 'NaN'
def filter_traj_by_missing_vessel_type(df):
    """Filter out complete vessel trajectories with a missing ship type."""
    # Convert 'NaN' strings to actual NaN values
    df['ship_type'] = df['ship_type'].replace('nan', pd.NA)
    return df.groupby('mmsi').filter(lambda x: x['ship_type'].notna().all())


# Set apropriate data types for each column
def set_column_data_type(df):
    """Set the data types used by the raw AIS dataset."""
    df['mmsi'] = df['mmsi'].astype(str)
    df['timestamp'] = pd.to_datetime(
        df['timestamp'], unit='s'
    )  # assuming the timestamp is in seconds
    df['sog'] = df['sog'].astype(float)
    df['cog'] = df['cog'].astype(float)
    if 'rot' in df.columns:
        df['rot'] = df['rot'].astype(float)
    df['ship_type'] = df['ship_type'].astype(str)
    return df


# Rename columns and rearrange them
def rename_and_rearrange_columns(df):
    """Rename AISViz columns and retain the raw columns used in the study."""
    df.rename(columns={'time': 'timestamp'}, inplace=True)  # time to timestamp
    df.rename(columns={'longitude': 'lon'}, inplace=True)  # longitude to lon
    df.rename(columns={'latitude': 'lat'}, inplace=True)  # latitude to lat

    df['ship_type'] = df['ship_type'].replace('nan', np.nan)

    # ROT was available in the raw data but was not used by the PINN.
    columns = [column for column in RAW_COLUMNS if column in df.columns]
    missing_columns = [column for column in COLS if column not in columns]
    if missing_columns:
        raise ValueError(f'Missing required AIS columns: {missing_columns}')

    df = df[columns]
    return df


# Prepare the raw AIS dataset
def prepare_raw_ais_data(ais_df):
    """Rename, type, sort, and classify the raw AIS messages."""
    ais_df = rename_and_rearrange_columns(ais_df)
    ais_df = set_column_data_type(ais_df)

    ais_df.sort_values(
        by=['mmsi', 'timestamp'], inplace=True
    )  # Sort by vessel and each vessel by timestamp

    ais_df = filter_traj_by_missing_vessel_type(
        ais_df
    )  # if vessel type is missing

    ais_df = replace_code_with_vessel_type(
        ais_df
    )  # Replace vessel codes with vessel type
    return ais_df


# Clean AIS messages
def clean_ais_messages(df, min_sog=0.5, max_sog=50.0):
    """Remove invalid, duplicate, and stationary AIS messages."""
    df = df.copy()

    # Wrap COG value to 0 - 360
    def COG_0_To_360(cog):
        cog = np.fmod(cog, 360.0)
        return np.where(cog < 0.0, cog + 360.0, cog)

    # Drop rows with all columns as NaN
    df.dropna(how='all', inplace=True)

    # Remove Invalid MMSIs (digits != 9)
    valid_mmsi = df['mmsi'].astype(str).str.strip().str.fullmatch(
        r'\d{9}', na=False
    )
    df = df[valid_mmsi]

    # Drop duplicate records based on MMSI and timestamp
    df.drop_duplicates(subset=['mmsi', 'timestamp'], inplace=True)

    # Filter trajectories with SOG >= 0.5
    df = df[(df['sog'] >= min_sog) & (df['sog'] <= max_sog)]

    # Wrap COG to range 0 - 360
    df['cog'] = COG_0_To_360(df['cog'])
    return df.sort_values(by=['mmsi', 'timestamp']).reset_index(drop=True)


# Set the minimum trajectory length to 300 AIS messages
def filter_trajectories_by_min_length(ais_df, min_length=300):
    """Retain MMSI trajectories with at least ``min_length`` messages."""
    return ais_df.groupby('mmsi').filter(lambda x: len(x) >= min_length)


# Perform all preprocessing steps on the AIS dataset
def preprocess_ais_data(
    ais_df,
    min_sog=0.5,
    max_sog=50.0,
    min_length=300,
    vessel_types=None,
):
    """Prepare and clean raw AIS data using the paper's preprocessing steps.

    Parameters
    ----------
    ais_df : pandas.DataFrame
        Raw AISViz messages.
    min_sog, max_sog : float
        Valid SOG interval in knots. Messages must have ``sog >= min_sog``.
    min_length : int
        Minimum number of messages required for each MMSI trajectory.
    vessel_types : iterable of str, optional
        Vessel types to retain, for example ``['cargo', 'tanker']``.
    """
    ais_df = prepare_raw_ais_data(ais_df)
    ais_df = clean_ais_messages(
        ais_df,
        min_sog=min_sog,
        max_sog=max_sog,
    )

    if vessel_types is not None:
        ais_df = ais_df[ais_df['ship_type'].isin(vessel_types)]

    ais_df = filter_trajectories_by_min_length(
        ais_df,
        min_length=min_length,
    )
    return ais_df.reset_index(drop=True)


# Segment vessel trips - Time Segmentation
# if time duration between points exceed threshold (1hr = 60mins)
def segment_trajectory_by_time(
    ais_df,
    time_threshold=60,
    min_segment_length=11,
):
    """Split one vessel trajectory when a time gap exceeds the threshold.

    This is the in-memory form of the original time-segmentation loop. It
    returns a list of DataFrames instead of writing each segment to a CSV file.
    """
    ais_df = ais_df.copy()
    ais_df['timestamp'] = pd.to_datetime(ais_df['timestamp'])
    ais_df.sort_values('timestamp', inplace=True)
    ais_df.reset_index(drop=True, inplace=True)

    segments = []
    lastIdx = 0

    for i in range(len(ais_df) - 1):
        dt = (
            ais_df.iloc[i + 1]['timestamp']
            - ais_df.iloc[i]['timestamp']
        ).total_seconds() / 60.0

        if dt > time_threshold:
            ndf = ais_df.iloc[lastIdx:i + 1].copy()
            lastIdx = i + 1
            if len(ndf) >= min_segment_length:
                segments.append(ndf)

    # Handle the last segment, or the complete trajectory when no gap exists.
    ndf = ais_df.iloc[lastIdx:].copy()
    if len(ndf) >= min_segment_length:
        segments.append(ndf)

    return segments


# -----------------------------------------------------------------------------
# Cubic spline and Cubic Hermite interpolation

# Cast type of each column in a dataset
def set_datatype(aisDF):
    """Cast each trajectory column to the original interpolation type."""
    for column in aisDF.columns:
        if column in ['mmsi', 'ship_type']:
            continue
        elif column == 'timestamp':
            aisDF[column] = aisDF[column].astype('datetime64[ns]')
        else:
            aisDF[column] = aisDF[column].astype('float')
    return aisDF


def do_interp(df, new_index, column, method):
    """Interpolate a trajectory column using the selected original method."""
    df_clean = df.dropna(subset=[column])  # Drop NaN values if any

    if method == 'cubic':  # Cubic Spline interpolation
        spline = CubicSpline(
            df_clean.index.astype(np.int64),
            df_clean[column],
            bc_type='clamped'
        )
        return spline(new_index.astype(np.int64))
    elif method == 'pchip':  # PCHIP (Cubic Hermite Spline) interpolation
        pchip = PchipInterpolator(
            df_clean.index.astype(np.int64),
            df_clean[column]
        )
        return pchip(new_index.astype(np.int64))
    else:
        raise ValueError(
            "Unsupported interpolation method. Choose either 'cubic' or "
            "'pchip'."
        )


def do_cog_interp(df, new_index, method):
    """Interpolate COG across the circular 0/360-degree boundary."""
    df_clean = df.dropna(subset=['cog']).copy()

    # Unwrap COG so that, for example, 359 to 1 degrees changes by 2 degrees.
    cog_radians = np.radians(df_clean['cog'].astype(float).to_numpy())
    df_clean['cog'] = np.degrees(np.unwrap(cog_radians))
    return do_interp(df_clean, new_index, 'cog', method)


def interpolate_trajectory(df, freq='2min', method='pchip'):
    """Interpolate one vessel trajectory without reading or writing files.

    The body retains the interpolation, SOG clamping, COG wrapping, and output
    construction from ``interp_traj`` in the original script.
    """
    # Cast data types and set index
    df = df[COLS].copy()
    df = set_datatype(df)
    df.sort_values('timestamp', inplace=True)
    df.drop_duplicates(subset=['timestamp'], inplace=True)
    df.set_index('timestamp', inplace=True)

    # Define new time range
    new_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq=freq
    )

    # Perform interpolation
    lat_interp = do_interp(df, new_index, 'lat', method)
    lon_interp = do_interp(df, new_index, 'lon', method)
    cog_interp = do_cog_interp(df, new_index, method)
    sog_interp = do_interp(df, new_index, 'sog', method)

    # CLAMPING SOG to its original [min, max] range
    sog_min = df['sog'].min()
    sog_max = df['sog'].max()
    sog_interp = np.clip(sog_interp, sog_min, sog_max)

    # WRAPPING COG to [0, 360)
    cog_interp = (cog_interp + 360) % 360

    # Create a new DataFrame with interpolated values
    df_interp = pd.DataFrame({
        'mmsi': df['mmsi'].iloc[0],
        'timestamp': new_index,
        'lat': lat_interp,
        'lon': lon_interp,
        'cog': cog_interp,
        'sog': sog_interp,
        'ship_type': df['ship_type'].iloc[0]
    })
    return df_interp


# -----------------------------------------------------------------------------
# Kinematic feature engineering
# https://www.movable-type.co.uk/scripts/latlong.html
# https://www.applanix.com/news/blog-course-heading-bearing/

# Utility Functions for Feature Engineering
def get_duration(aisDF):
    """Calculate duration between trajectory points in seconds."""
    tDiff = np.diff(pd.to_datetime(aisDF.index)) / np.timedelta64(1, 's')
    tDiff = np.insert(
        tDiff.astype(np.float64), 0, 0.0
    )  # Insert '0.0' for the first record
    return tDiff


# New function to convert SOG from knots to m/s in the dataset
def convert_sog_to_ms(aisDF):
    """Convert SOG from knots to metres per second."""
    aisDF['sog'] = aisDF['sog'] * 0.514444  # 1 knot = 0.514444 m/s
    return aisDF


# Updated acceleration calculation (assuming SOG already in m/s)
def get_acceleration(aisDF):
    """Calculate acceleration of the vessel in metres per second squared."""
    sogDiff = np.diff(aisDF.sog)  # No conversion needed now
    acc = np.insert(sogDiff, 0, 0.0) / aisDF.duration
    return np.nan_to_num(acc)  # NaN replaced with '0.0'


def get_cog_rate(aisDF):
    """Calculate rate of change of COG in degrees per second."""
    # Use the shortest signed difference across the 0/360-degree boundary.
    cogDiff = (np.diff(aisDF.cog) + 180.0) % 360.0 - 180.0
    aSpeed = np.insert(cogDiff, 0, 0.0) / aisDF.duration
    return np.nan_to_num(aSpeed)


# Do feature engineering on vessel(mmsi) trajectories
def add_kinematic_features(aisDF):
    """Add the paper's acceleration and COG-rate features to a trajectory.

    This is the reusable DataFrame portion of the original ``FE`` function.
    SOG is expected in knots and is returned in metres per second.
    """
    aisDF = aisDF[COLS].copy()
    if 'timestamp' in aisDF.columns:
        aisDF.set_index('timestamp', inplace=True)

    aisDF = aisDF.assign(duration=get_duration(aisDF))
    aisDF = convert_sog_to_ms(aisDF)  # Convert SOG to m/s
    aisDF = aisDF.assign(acceleration=get_acceleration(aisDF))
    aisDF = aisDF.assign(cograte=get_cog_rate(aisDF))
    aisDF = aisDF.assign(timestamp=aisDF.index)

    aisDF.drop(['duration'], axis=1, inplace=True)

    # Reorder columns
    cols_order = [
        'mmsi', 'timestamp', 'lat', 'lon', 'sog', 'cog',
        'acceleration', 'cograte', 'ship_type'
    ]
    return aisDF[cols_order]
