"""
loader.py — Data Loading & Cleaning Module
===========================================
This is the FOUNDATION of our entire dashboard.
Every other module (analysis.py, visualizer.py, app.py) depends on this file.

What this file does:
1. Fetches real historical weather data from the Open-Meteo Archive API
2. Caches it to a CSV so we don't hit the API every time
3. Cleans the raw data (fills missing values, renames columns)
4. Adds useful derived columns (month, season, year, etc.)
5. Detects outliers using the IQR method
6. Provides data quality summaries
7. Aggregates data to different time frequencies (daily/weekly/monthly)
"""

# ──────────────────────────────────────────────
# IMPORTS — Libraries we need
# ──────────────────────────────────────────────

import os               # For file path operations (checking if CSV exists)
import requests          # For making HTTP calls to the Open-Meteo API
import pandas as pd      # Our main data manipulation library (DataFrames)
import numpy as np       # Numerical operations (NaN handling, math)
import streamlit as st   # We use st.cache_data to cache expensive operations
from datetime import datetime  # For date parsing and manipulation


# ──────────────────────────────────────────────
# CITIES CONFIGURATION
# ──────────────────────────────────────────────
# This dictionary holds the latitude/longitude for each city we want data for.
# The Open-Meteo API needs coordinates, not city names.
# We picked 7 cities across 7 countries with DIFFERENT climates
# so our analysis shows interesting contrasts.

CITIES = {
    "Karachi": {
        "lat": 24.86,        # Latitude  (how far north/south from equator)
        "lon": 67.01,        # Longitude (how far east/west from prime meridian)
        "country": "Pakistan"
    },
    "Mumbai": {
        "lat": 19.08,
        "lon": 72.88,
        "country": "India"
    },
    "Riyadh": {
        "lat": 24.71,
        "lon": 46.68,
        "country": "Saudi Arabia"
    },
    "Dubai": {
        "lat": 25.28,
        "lon": 55.30,
        "country": "UAE"
    },
    "New York": {
        "lat": 40.71,
        "lon": -74.01,       # Negative = West of prime meridian
        "country": "USA"
    },
    "Auckland": {
        "lat": -36.85,       # Negative = South of equator
        "lon": 174.76,
        "country": "New Zealand"
    },
    "Sydney": {
        "lat": -33.87,
        "lon": 151.21,
        "country": "Australia"
    },
}

# The weather variables we want from the API.
# These are the exact parameter names the Open-Meteo API expects.
# We request them as "hourly" variables (one value per hour).
HOURLY_VARIABLES = [
    "temperature_2m",          # Air temperature at 2 meters height (°C)
    "apparent_temperature",    # "Feels like" temperature (accounts for wind/humidity)
    "relative_humidity_2m",    # Humidity percentage at 2 meters height
    "precipitation",           # Rain/snow in millimeters
    "wind_speed_10m",          # Wind speed at 10 meters height (km/h)
    "wind_direction_10m",      # Wind direction in degrees (0=North, 90=East, etc.)
    "surface_pressure",        # Atmospheric pressure in hPa (hectopascals)
]

# Human-friendly names for display in the dashboard.
# The API returns ugly column names like "temperature_2m",
# so we rename them to something a user can understand.
COLUMN_RENAME_MAP = {
    "temperature_2m": "Temperature (°C)",
    "apparent_temperature": "Apparent Temperature (°C)",
    "relative_humidity_2m": "Humidity (%)",
    "precipitation": "Precipitation (mm)",
    "wind_speed_10m": "Wind Speed (km/h)",
    "wind_direction_10m": "Wind Direction (°)",
    "surface_pressure": "Pressure (hPa)",
}

# Which season does each month belong to?
# We use this to add a "Season" column for seasonal analysis.
# Note: This is Northern Hemisphere convention.
# For Auckland/Sydney (Southern Hemisphere), seasons are actually flipped,
# but we keep it consistent for simplicity in the dashboard.
MONTH_TO_SEASON = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring",  4: "Spring",  5: "Spring",
    6: "Summer",  7: "Summer",  8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn",
}

# Default path where we cache the downloaded data
DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),  # Go up from utils/ to weather_dashboard/
    "data",
    "weather_data.csv"
)


# ──────────────────────────────────────────────
# FUNCTION 1: fetch_from_api
# ──────────────────────────────────────────────
# PURPOSE: Hit the Open-Meteo Archive API for each city, download hourly
#          weather data, and combine everything into one big DataFrame.
#
# WHY WE NEED IT: This is how we get our raw data. Without this function,
#                 we'd have no data to analyze or visualize.

def fetch_from_api(
    start_date=None,
    end_date=None,
):
    """
    Fetch historical hourly weather data from Open-Meteo Archive API.

    Parameters:
    -----------
    start_date : str | None
        Start date in "YYYY-MM-DD" format. We default to 2020-01-01
        to get 5 years of data (good balance of richness vs download speed).
    end_date : str | None
        End date in "YYYY-MM-DD" format.

    Returns:
    --------
    pd.DataFrame
        A single DataFrame with ALL cities' data stacked together.
        Each row = one hour of weather data for one city.
    """

    # The base URL for the Open-Meteo historical weather API.
    # This is a FREE API — no API key needed!
    api_url = "https://archive-api.open-meteo.com/v1/archive"

    # We'll collect each city's data in this list, then combine at the end
    all_city_frames = []

    # Loop through each city in our CITIES dictionary
    for city_name, city_info in CITIES.items():

        # Print progress so we know which city is being fetched
        # (this shows up in the Streamlit terminal/logs)
        print(f"  Fetching data for {city_name}...")

        # Build the query parameters for the API request.
        # The API docs tell us exactly what parameters it expects:
        # https://open-meteo.com/en/docs/historical-weather-api
        params = {
            "latitude": city_info["lat"],           # City's latitude
            "longitude": city_info["lon"],           # City's longitude
            "start_date": start_date if start_date else (datetime.utcnow() - pd.Timedelta(days=30)).strftime("%Y-%m-%d"),  # 30‑day window default
            "end_date": end_date if end_date else datetime.utcnow().strftime("%Y-%m-%d"),
            "hourly": ",".join(HOURLY_VARIABLES),    # Join variable names with commas
            # e.g. "temperature_2m,apparent_temperature,..."
            "timezone": "auto",                      # Let API pick timezone based on location
        }

        # Make the HTTP GET request to the API
        # requests.get() sends a GET request and returns the response
        response = requests.get(api_url, params=params, timeout=60)

        # Check if the request was successful (status code 200 = OK)
        # If not (e.g. 404, 500), this raises an exception with the error details
        response.raise_for_status()

        # Parse the JSON response into a Python dictionary
        data = response.json()

        # The API returns data in this structure:
        # {
        #   "hourly": {
        #       "time": ["2020-01-01T00:00", "2020-01-01T01:00", ...],
        #       "temperature_2m": [15.2, 14.8, ...],
        #       "humidity_2m": [72, 75, ...],
        #       ...
        #   }
        # }
        # We extract the "hourly" section and convert it into a DataFrame
        hourly_data = data["hourly"]

        # pd.DataFrame() converts the dictionary into a table:
        # Each key becomes a column, each list element becomes a row
        df_city = pd.DataFrame(hourly_data)

        # Add a "City" column so we know which city each row belongs to
        # (since we're combining all cities into one DataFrame)
        df_city["City"] = city_name

        # Add a "Country" column for extra context
        df_city["Country"] = city_info["country"]

        # Append this city's DataFrame to our collection list
        all_city_frames.append(df_city)

    # pd.concat() stacks all the individual city DataFrames vertically
    # ignore_index=True resets the row numbers (0, 1, 2, ...) instead of
    # keeping the original indices from each city's DataFrame
    combined_df = pd.concat(all_city_frames, ignore_index=True)

    return combined_df


# ──────────────────────────────────────────────
# FUNCTION 2: load_data
# ──────────────────────────────────────────────
# PURPOSE: Smart data loader — checks if we already have cached data.
#          If yes, load from CSV (fast). If no, fetch from API (slow, ~30 seconds).
#
# WHY WE NEED IT: We don't want to hit the API every time the user
#                 refreshes the dashboard. The CSV cache makes it instant.
#
# The @st.cache_data decorator tells Streamlit:
#   "Run this function ONCE, save the result in memory.
#    Next time someone calls it with the same arguments, return
#    the saved result instead of running the function again."
# This is critical for performance — without it, the data would reload
# on every single user interaction.

@st.cache_data(show_spinner="Loading weather data...")
def load_data(start_date=None, end_date=None, path=None):
    """
    Load weather data from cache (CSV) or fetch from API if not cached.

    Parameters:
    -----------
    start_date : str | None
        Start date in "YYYY-MM-DD" format. If None, defaults to 30‑day window ending today.
    end_date : str | None
        End date in "YYYY-MM-DD" format. If None, defaults to today.
    path : str | None
        Optional explicit CSV path. If None, a path is generated based on the date range.

    Returns:
    --------
    pd.DataFrame
        Cleaned and processed weather data ready for analysis.
    """

    # Resolve default dates if not supplied
    if start_date is None:
        start_date = (datetime.utcnow() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    if end_date is None:
        end_date = datetime.utcnow().strftime("%Y-%m-%d")

    # Build a deterministic cache filename based on the range
    if path is None:
        filename = f"weather_{start_date}_to_{end_date}.csv"
        path = os.path.join(os.path.dirname(DEFAULT_DATA_PATH), filename)

    # Check if the CSV file already exists on disk
    if os.path.exists(path):
        print(f"Loading cached data from {path}...")
        df = pd.read_csv(path)
    else:
        print("No cached data found. Fetching from Open-Meteo API...")
        df = fetch_from_api(start_date=start_date, end_date=end_date)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        print(f"Data cached to {path}")

    df = clean_data(df)
    return df


# ──────────────────────────────────────────────
# FUNCTION 3: clean_data
# ──────────────────────────────────────────────
# PURPOSE: Transform raw API data into a clean, analysis-ready DataFrame.
#
# WHY WE NEED IT: Raw data from any API is messy:
#   - Column names are ugly ("temperature_2m" instead of "Temperature (°C)")
#   - Dates are strings instead of proper datetime objects
#   - Some hours might have missing values (sensor glitches, API gaps)
#   - We need derived columns (month, season) for grouping in charts
#
# This function fixes ALL of those issues in one pass.

def clean_data(df):
    """
    Clean and enrich the raw weather DataFrame.

    Steps:
    1. Parse the 'time' column into proper datetime objects
    2. Rename API column names to human-friendly names
    3. Fill missing values using forward-fill + interpolation
    4. Add derived columns: month, year, season, day_of_week, day_of_year

    Parameters:
    -----------
    df : pd.DataFrame
        Raw DataFrame from API or CSV.

    Returns:
    --------
    pd.DataFrame
        Cleaned DataFrame ready for analysis.
    """

    # Make a copy so we don't accidentally modify the original DataFrame.
    # In Python, DataFrames are "mutable" — if we modify df directly,
    # it would change the original data everywhere it's referenced.
    df = df.copy()

    # ── Step 1: Parse the 'time' column ──
    # The API gives us time as strings like "2020-01-01T00:00".
    # pd.to_datetime() converts them into proper datetime objects,
    # which lets us do time-based operations (filtering by month, resampling, etc.)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])

    # ── Step 2: Rename columns to human-friendly names ──
    # We only rename columns that actually exist in our data
    # (in case the API didn't return some variables)
    rename_dict = {
        old: new for old, new in COLUMN_RENAME_MAP.items()
        if old in df.columns
    }
    df.rename(columns=rename_dict, inplace=True)
    #   inplace=True means "modify this DataFrame directly"
    #   instead of creating a new one (saves memory)

    # ── Step 3: Fill missing values ──
    # Weather APIs sometimes have gaps (e.g., a sensor went offline for an hour).
    # We use TWO strategies to fill these gaps:

    # Get the list of numeric columns (temperature, humidity, etc.)
    # We only fill numeric columns — we don't want to fill "City" or "Country"
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Strategy A: Forward-fill (ffill)
    # Takes the PREVIOUS valid value and copies it forward.
    # Example: [25, NaN, NaN, 26] → [25, 25, 25, 26]
    # This works well for weather because temperature doesn't change drastically
    # from one hour to the next.
    df[numeric_cols] = df[numeric_cols].ffill()

    # Strategy B: Linear interpolation (for any remaining gaps)
    # If forward-fill couldn't fix it (e.g., the FIRST row is NaN),
    # interpolation estimates the value based on surrounding values.
    # Example: [NaN, 20, NaN, 24] → [20, 20, 22, 24]  (linear estimate)
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear")

    # Strategy C: Fill any STILL remaining NaNs with 0
    # This catches edge cases where both ffill and interpolation couldn't help
    # (e.g., an entire column is NaN — very rare but possible)
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # ── Step 4: Add derived time columns ──
    # These columns let us GROUP and FILTER data in interesting ways:
    # "Show me average temperature by month" or "Compare weekdays vs weekends"
    if "time" in df.columns:
        df["Month"] = df["time"].dt.month          # 1-12
        df["Year"] = df["time"].dt.year             # e.g. 2020, 2021
        df["Day of Week"] = df["time"].dt.day_name()  # "Monday", "Tuesday", etc.
        df["Day of Year"] = df["time"].dt.dayofyear   # 1-366
        df["Hour"] = df["time"].dt.hour                # 0-23

        # Map each month number to its season name using our MONTH_TO_SEASON dict
        # .map() replaces each value using the dictionary as a lookup table
        # e.g., month 1 → "Winter", month 6 → "Summer"
        df["Season"] = df["Month"].map(MONTH_TO_SEASON)

    return df


# ──────────────────────────────────────────────
# FUNCTION 4: remove_outliers_iqr
# ──────────────────────────────────────────────
# PURPOSE: Detect outlier values using the IQR (Interquartile Range) method.
#
# WHY WE NEED IT: Outliers are extreme values that can distort our analysis.
#   For example, if a temperature sensor glitches and reports 500°C,
#   it would skew our average. We FLAG these but DON'T delete them,
#   because they're interesting to show in our anomaly visualizations.
#
# HOW IQR WORKS:
#   1. Sort all values and find Q1 (25th percentile) and Q3 (75th percentile)
#   2. IQR = Q3 - Q1 (the range of the "middle 50%" of data)
#   3. Any value below Q1 - 1.5*IQR or above Q3 + 1.5*IQR is an outlier
#
#   Example with temperatures: [10, 12, 13, 14, 15, 15, 16, 17, 18, 50]
#   Q1=12.75, Q3=17.25, IQR=4.5
#   Lower bound = 12.75 - 6.75 = 6.0
#   Upper bound = 17.25 + 6.75 = 24.0
#   → 50 is flagged as an outlier (above 24.0)

def remove_outliers_iqr(df, column):
    """
    Flag outliers in a specific column using the IQR method.

    Parameters:
    -----------
    df : pd.DataFrame
        The weather DataFrame.
    column : str
        Which column to check for outliers (e.g., "Temperature (°C)").

    Returns:
    --------
    tuple: (cleaned_df, outlier_count)
        - cleaned_df: DataFrame with an added "is_outlier" column (True/False)
        - outlier_count: How many outliers were found
    """

    df = df.copy()

    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    # .quantile(0.25) gives us the value below which 25% of data falls
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)

    # IQR = the spread of the middle 50% of data
    iqr = q3 - q1

    # Define the "fence" — anything outside this range is an outlier
    # The 1.5 multiplier is the standard statistical convention
    # (invented by John Tukey, the father of box plots)
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Create a boolean column: True = outlier, False = normal
    # The | means "OR" — a value is an outlier if it's below lower OR above upper
    df["is_outlier"] = (df[column] < lower_bound) | (df[column] > upper_bound)

    # Count how many outliers we found
    outlier_count = df["is_outlier"].sum()

    return df, outlier_count


# ──────────────────────────────────────────────
# FUNCTION 5: get_data_quality_summary
# ──────────────────────────────────────────────
# PURPOSE: Generate a quick "health report" of our dataset.
#
# WHY WE NEED IT: Before doing any analysis, it's good practice to know:
#   - How many rows do we have?
#   - Are there missing values? How many?
#   - Are there outliers? In which columns?
# We display this in the sidebar of our dashboard so the user
# can see the data quality at a glance.

def get_data_quality_summary(df):
    """
    Generate a summary of data quality metrics.

    Parameters:
    -----------
    df : pd.DataFrame
        The weather DataFrame.

    Returns:
    --------
    dict: Contains:
        - total_rows: Total number of rows in the dataset
        - missing_pct: Dictionary of {column: missing_percentage}
        - outlier_counts: Dictionary of {column: number_of_outliers}
    """

    summary = {}

    # Total number of rows
    # len(df) returns the number of rows in the DataFrame
    summary["total_rows"] = len(df)

    # Missing value percentages per column
    # df.isnull() creates a True/False table (True where values are missing)
    # .sum() counts the Trues per column
    # / len(df) converts to a fraction
    # * 100 converts to percentage
    # .to_dict() converts the result from a pandas Series to a plain dictionary
    summary["missing_pct"] = (
        (df.isnull().sum() / len(df) * 100)
        .round(2)        # Round to 2 decimal places
        .to_dict()
    )

    # Outlier counts for each numeric column
    # We reuse our remove_outliers_iqr function for each column
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    outlier_counts = {}
    for col in numeric_cols:
        _, count = remove_outliers_iqr(df, col)
        #  ^ We don't need the modified DataFrame (first return value),
        #    only the count (second return value), so we use _ as a throwaway
        outlier_counts[col] = count
    summary["outlier_counts"] = outlier_counts

    return summary


# ──────────────────────────────────────────────
# FUNCTION 6: aggregate_data
# ──────────────────────────────────────────────
# PURPOSE: Resample hourly data to daily, weekly, or monthly summaries.
#
# WHY WE NEED IT: Hourly data is great for detail, but for most charts
#   we want daily or monthly averages. Looking at 43,800 hourly data points
#   per city (5 years × 365 days × 24 hours) would make charts unreadable.
#
# HOW RESAMPLING WORKS:
#   Imagine you have hourly temperatures for January 1st:
#   [10, 11, 12, 15, 18, 20, 22, 21, 19, 17, 15, 13, 12, 11, 10, 9, 8, 9, 10, 11, 12, 13, 14, 15]
#
#   Daily resample with "mean" → 13.6°C (average of all 24 hours)
#   Daily resample with "sum"  → would be wrong for temperature (but correct for precipitation!)
#
# That's why we use DIFFERENT aggregation methods for different variables:
#   - Temperature, humidity, wind → MEAN (average makes sense)
#   - Precipitation → SUM (total rainfall for the day makes sense)

def aggregate_data(df, freq="D"):
    """
    Resample hourly data to a coarser time frequency.

    Parameters:
    -----------
    df : pd.DataFrame
        The weather DataFrame with a 'time' column.
    freq : str
        Pandas frequency string:
        - "D" = Daily
        - "W" = Weekly
        - "ME" = Month-End (monthly)

    Returns:
    --------
    pd.DataFrame
        Aggregated DataFrame with one row per time period per city.
    """

    # If there's no 'time' column, we can't resample — return as-is
    if "time" not in df.columns:
        return df

    # We need to aggregate SEPARATELY for each city.
    # If we resampled all cities together, their values would get mixed up!
    # groupby("City") splits the data into separate groups, one per city.
    aggregated_frames = []

    for city_name, city_df in df.groupby("City"):

        # Set the 'time' column as the index (row label).
        # Pandas .resample() requires a DatetimeIndex to work.
        city_df = city_df.set_index("time")

        # Define which columns to aggregate with mean vs sum
        # Mean columns: temperatures, humidity, wind, pressure
        mean_cols = [
            col for col in city_df.columns
            if col in [
                "Temperature (°C)", "Apparent Temperature (°C)",
                "Humidity (%)", "Wind Speed (km/h)",
                "Wind Direction (°)", "Pressure (hPa)",
            ]
        ]

        # Sum columns: only precipitation (we want total rain, not average rain)
        sum_cols = [
            col for col in city_df.columns
            if col in ["Precipitation (mm)"]
        ]

        # Build the aggregation dictionary
        # This tells resample: "for temperature, take the mean; for precipitation, take the sum"
        agg_dict = {}
        for col in mean_cols:
            agg_dict[col] = "mean"
        for col in sum_cols:
            agg_dict[col] = "sum"

        # Only proceed if we have columns to aggregate
        if not agg_dict:
            continue

        # .resample(freq) groups the data by time periods (daily/weekly/monthly)
        # .agg(agg_dict) applies the appropriate function to each column
        resampled = city_df.resample(freq).agg(agg_dict)

        # Round to 2 decimal places for cleanliness
        resampled = resampled.round(2)

        # Add the city name back (it was lost when we grouped)
        resampled["City"] = city_name

        # Get the country name from our CITIES config
        resampled["Country"] = CITIES.get(city_name, {}).get("country", "")

        # Reset index so 'time' becomes a regular column again
        # (some of our chart functions expect 'time' as a column, not an index)
        resampled = resampled.reset_index()

        # Re-add derived time columns that were lost during resampling
        resampled["Month"] = resampled["time"].dt.month
        resampled["Year"] = resampled["time"].dt.year
        resampled["Season"] = resampled["Month"].map(MONTH_TO_SEASON)
        resampled["Day of Week"] = resampled["time"].dt.day_name()
        resampled["Day of Year"] = resampled["time"].dt.dayofyear

        aggregated_frames.append(resampled)

    # Combine all cities back into one DataFrame
    if aggregated_frames:
        result = pd.concat(aggregated_frames, ignore_index=True)
    else:
        result = df  # Fallback: return original if aggregation failed

    return result
