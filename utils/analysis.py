"""
analysis.py — Statistical Analysis Module
==========================================
This module provides all the statistical methods used by the dashboard.
It sits BETWEEN loader.py (which gives us clean data) and visualizer.py
(which turns our analysis results into charts).

What this file does:
1. Computes summary statistics (mean, max, min, std)
2. Calculates rolling averages and rolling standard deviations
3. Detects anomalies using z-scores
4. Fits linear trend lines using linear regression
5. Computes correlation matrices (Pearson & Spearman)
6. Decomposes time series into trend + seasonal + residual
7. Runs ARIMA forecasting with error metrics
8. Auto-generates human-readable text insights
"""

# ──────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────

import numpy as np                      # Numerical math (mean, std, z-scores)
import pandas as pd                     # DataFrame operations
from scipy import stats                 # Linear regression (linregress)
from statsmodels.tsa.seasonal import seasonal_decompose  # STL decomposition
from statsmodels.tsa.arima.model import ARIMA            # ARIMA forecasting
from sklearn.metrics import mean_absolute_error, mean_squared_error  # Forecast accuracy


# ──────────────────────────────────────────────
# FUNCTION 1: compute_summary_stats
# ──────────────────────────────────────────────
# PURPOSE: Calculate the 5 key summary numbers for any numeric series.
#          These appear in the metric cards at the top of the dashboard.
#
# WHY WE NEED IT: Summary stats give users an instant overview:
#   "What's the average temperature? What was the hottest day?
#    How much does it vary?" — all answered in one function call.

def compute_summary_stats(series):
    """
    Compute basic descriptive statistics for a numeric series.

    Parameters:
    -----------
    series : pd.Series
        A column of numeric values (e.g., temperature readings).

    Returns:
    --------
    dict: Keys → 'mean', 'max', 'min', 'std', 'median', 'range', 'count'
    """

    # .dropna() removes any NaN values before computing stats.
    # If we don't do this, NaNs would propagate and make all results NaN.
    clean = series.dropna()

    return {
        "mean": round(clean.mean(), 2),        # Average value
        "max": round(clean.max(), 2),          # Highest value
        "min": round(clean.min(), 2),          # Lowest value
        "std": round(clean.std(), 2),          # Standard deviation (spread)
        "median": round(clean.median(), 2),    # Middle value (50th percentile)
        "range": round(clean.max() - clean.min(), 2),  # Max minus min
        "count": int(clean.count()),           # Number of non-null data points
    }


# ──────────────────────────────────────────────
# FUNCTION 2: rolling_statistics
# ──────────────────────────────────────────────
# PURPOSE: Calculate rolling (moving) averages and rolling standard deviations.
#
# WHY WE NEED IT: Raw weather data is noisy — temperatures jump up and down
#   hour to hour. A 7-day rolling average smooths out daily noise and reveals
#   the underlying trend. A 30-day rolling average smooths even more,
#   showing seasonal patterns.
#
# HOW ROLLING WORKS:
#   Imagine daily temperatures: [20, 22, 19, 21, 23, 18, 24]
#   7-day rolling mean at day 7 = average(20,22,19,21,23,18,24) = 21.0
#   The "window" slides forward one day at a time, always averaging
#   the most recent 7 values.
#
#   min_periods=1 means: "Don't wait until you have a full window.
#   For the first 6 days, just average whatever you have so far."
#   Without this, the first 6 values would be NaN.

def rolling_statistics(series, windows=None):
    """
    Compute rolling mean and rolling std for given window sizes.

    Parameters:
    -----------
    series : pd.Series
        A column of numeric values.
    windows : list[int] | None
        Window sizes in number of data points.
        Default is [7, 30] (7-day and 30-day).

    Returns:
    --------
    dict: Keys are window sizes, values are dicts with 'mean' and 'std' Series.
          Example: {7: {'mean': pd.Series, 'std': pd.Series}, 30: {...}}
    """

    if windows is None:
        windows = [7, 30]

    results = {}
    for w in windows:
        results[w] = {
            # .rolling(w) creates a sliding window of size w
            # .mean() averages the values in each window position
            "mean": series.rolling(window=w, min_periods=1).mean(),
            # .std() computes the standard deviation in each window
            # This tells us how volatile the values are within that window
            "std": series.rolling(window=w, min_periods=1).std(),
        }

    return results


# ──────────────────────────────────────────────
# FUNCTION 3: detect_anomalies_zscore
# ──────────────────────────────────────────────
# PURPOSE: Find data points that are "abnormally" far from the average.
#
# WHY WE NEED IT: Anomalies are the most interesting data points!
#   "Which day was unusually hot?" "When did we get a freak rainstorm?"
#   These show up as red dots on our time series chart.
#
# HOW Z-SCORES WORK:
#   A z-score tells you "how many standard deviations away from the mean
#   is this value?"
#
#   z = (value - mean) / std_deviation
#
#   Example: If mean temp = 25°C and std = 5°C:
#     - 30°C → z = (30-25)/5 = 1.0   (1 std above mean — normal)
#     - 40°C → z = (40-25)/5 = 3.0   (3 stds above mean — ANOMALY!)
#
#   We use |z| > threshold (default 2) to flag anomalies.
#   With threshold=2, roughly 5% of data points get flagged.

def detect_anomalies_zscore(series, threshold=2):
    """
    Detect anomalies using z-score method.

    Parameters:
    -----------
    series : pd.Series
        A column of numeric values.
    threshold : float
        Z-score threshold. Values with |z| > threshold are anomalies.
        Default is 2 (≈ 5% of normally distributed data).

    Returns:
    --------
    pd.Series (bool): True where the value is an anomaly, False otherwise.
    """

    clean = series.dropna()

    # Calculate mean and standard deviation of the series
    mean = clean.mean()
    std = clean.std()

    # Avoid division by zero — if all values are identical, std=0
    if std == 0:
        return pd.Series(False, index=series.index)

    # Compute z-scores for every value
    # np.abs() takes the absolute value so we catch both unusually HIGH
    # and unusually LOW values
    z_scores = np.abs((series - mean) / std)

    # Return a boolean mask: True = anomaly, False = normal
    return z_scores > threshold


# ──────────────────────────────────────────────
# FUNCTION 4: linear_trend
# ──────────────────────────────────────────────
# PURPOSE: Fit a straight line through the data to see if there's
#          an upward or downward trend over time.
#
# WHY WE NEED IT: "Is temperature increasing over the years?"
#   A linear trend line answers this visually and numerically.
#   The slope tells you HOW FAST it's changing, and R² tells you
#   how well a straight line fits the data.
#
# HOW LINEAR REGRESSION WORKS:
#   We fit y = slope * x + intercept
#   where x = position in time (0, 1, 2, ...) and y = the weather value.
#
#   scipy.stats.linregress() uses the "least squares" method —
#   it finds the line that minimizes the total squared distance
#   from all points to the line.

def linear_trend(series):
    """
    Fit a linear regression trend line to a numeric series.

    Parameters:
    -----------
    series : pd.Series
        A column of numeric values (time-ordered).

    Returns:
    --------
    dict: Keys → 'slope', 'intercept', 'r_squared', 'p_value', 'trend_line'
        - slope: How much the value changes per time step
        - intercept: The starting value of the trend line
        - r_squared: How well the line fits (0=terrible, 1=perfect)
        - p_value: Statistical significance (< 0.05 = significant)
        - trend_line: pd.Series of fitted values (same length as input)
    """

    # Drop NaN values — regression can't handle missing data
    clean = series.dropna()

    if len(clean) < 2:
        # Need at least 2 points to draw a line
        return {
            "slope": 0, "intercept": 0, "r_squared": 0,
            "p_value": 1, "trend_line": series,
        }

    # Create x-axis values: 0, 1, 2, 3, ..., n-1
    # We use position numbers instead of actual dates because
    # linregress needs numeric x values
    x = np.arange(len(clean))
    y = clean.values

    # scipy.stats.linregress fits the line and returns:
    #   slope: rise/run (how steep the line is)
    #   intercept: where the line crosses y-axis
    #   rvalue: correlation coefficient (-1 to +1)
    #   pvalue: probability that the slope is actually zero
    #   stderr: standard error of the slope estimate
    slope, intercept, rvalue, pvalue, stderr = stats.linregress(x, y)

    # Generate the trend line values: y = slope * x + intercept
    # We create a Series with the ORIGINAL index so it aligns
    # with the input data for plotting
    trend_values = slope * np.arange(len(series)) + intercept
    trend_line = pd.Series(trend_values, index=series.index)

    return {
        "slope": round(slope, 6),
        "intercept": round(intercept, 2),
        "r_squared": round(rvalue ** 2, 4),    # R² = correlation² (always positive)
        "p_value": round(pvalue, 6),
        "trend_line": trend_line,
    }


# ──────────────────────────────────────────────
# FUNCTION 5: correlation_matrix
# ──────────────────────────────────────────────
# PURPOSE: Calculate how strongly different weather variables are
#          related to each other.
#
# WHY WE NEED IT: "Does higher humidity mean more rain?"
#   "Does wind speed affect temperature?" Correlation analysis
#   answers these questions with numbers.
#
# HOW CORRELATION WORKS:
#   Pearson correlation ranges from -1 to +1:
#     +1 = perfect positive relationship (one goes up, other goes up)
#      0 = no relationship
#     -1 = perfect negative relationship (one goes up, other goes down)
#
#   Spearman correlation is similar but works on RANKS instead of raw values.
#   It catches non-linear relationships that Pearson misses.
#   Example: If temperature and ice cream sales both increase but not at
#   a constant rate, Spearman might give a higher correlation than Pearson.

def correlation_matrix(df, columns=None, method="pearson"):
    """
    Compute a correlation matrix between selected numeric columns.

    Parameters:
    -----------
    df : pd.DataFrame
        The weather DataFrame.
    columns : list[str] | None
        Which columns to include. If None, uses all numeric columns.
    method : str
        'pearson' (default) or 'spearman'.

    Returns:
    --------
    pd.DataFrame: Square correlation matrix (values between -1 and +1).
    """

    if columns is None:
        # Auto-select all numeric columns
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

        # Remove non-weather columns that would clutter the heatmap
        exclude = {"Month", "Year", "Day of Year", "Hour"}
        columns = [c for c in columns if c not in exclude]

    # .corr(method=...) computes pairwise correlation between all columns
    # The result is a square DataFrame where row i, column j = correlation(i, j)
    return df[columns].corr(method=method).round(3)


# ──────────────────────────────────────────────
# FUNCTION 6: seasonal_decompose_series
# ──────────────────────────────────────────────
# PURPOSE: Break a time series into three components:
#          Trend + Seasonal Pattern + Residual (noise).
#
# WHY WE NEED IT: "Is the temperature rising because of a long-term trend,
#   or just because summer is coming?" Decomposition separates these effects
#   so we can see each one independently.
#
# HOW STL DECOMPOSITION WORKS:
#   Given a time series like daily temperature:
#     Observed = Trend + Seasonal + Residual
#
#   - Trend: The long-term direction (smoothed, like a 365-day moving average)
#   - Seasonal: The repeating yearly pattern (hot summers, cold winters)
#   - Residual: Whatever's left after removing trend and seasonality (random noise)
#
#   The 'period' parameter tells the algorithm "the pattern repeats every N points".
#   For daily data, period=365 (yearly cycle). For monthly, period=12.

def seasonal_decompose_series(series, period=None):
    """
    Perform seasonal decomposition on a time series.

    Parameters:
    -----------
    series : pd.Series
        Time-indexed numeric series (must have a DatetimeIndex or
        be regularly spaced).
    period : int | None
        The period of the seasonal component. If None, we try to infer it:
        - Daily data → period=365 (yearly cycle)
        - Monthly data → period=12

    Returns:
    --------
    DecomposeResult | None
        Object with .trend, .seasonal, .resid, .observed attributes.
        Returns None if decomposition fails (e.g., not enough data).
    """

    # Drop NaN values — decomposition needs a clean series
    clean = series.dropna()

    # Need at least 2 full cycles of data for meaningful decomposition
    if period is None:
        # Try to guess the period from the data frequency
        n = len(clean)
        if n >= 730:          # 2+ years of daily data
            period = 365
        elif n >= 60:         # 2+ years of monthly data OR 2+ months of daily
            period = 30
        elif n >= 14:         # At least 2 weeks
            period = 7
        else:
            return None       # Not enough data

    # Need at least 2 full periods for the algorithm to work
    if len(clean) < 2 * period:
        # Reduce period if we don't have enough data
        period = max(2, len(clean) // 2)

    try:
        # model='additive' means: Observed = Trend + Seasonal + Residual
        # (as opposed to 'multiplicative': Observed = Trend × Seasonal × Residual)
        # Additive is appropriate when the seasonal swing doesn't grow over time.
        # extrapolate_trend='freq' handles edge NaN values in the trend component.
        result = seasonal_decompose(
            clean,
            model="additive",
            period=period,
            extrapolate_trend="freq",
        )
        return result
    except Exception:
        # If decomposition fails for any reason (weird data, not enough points),
        # return None gracefully instead of crashing the dashboard
        return None


# ──────────────────────────────────────────────
# FUNCTION 7: arima_forecast
# ──────────────────────────────────────────────
# PURPOSE: Predict future weather values using the ARIMA model.
#
# WHY WE NEED IT: "What will the temperature be next week?"
#   ARIMA is a classic time-series forecasting model that learns
#   from past patterns to predict future values.
#
# HOW ARIMA WORKS (simplified):
#   ARIMA(p, d, q) has three components:
#     p = AutoRegressive: "Use the last p values to predict the next one"
#     d = Integrated: "Difference the data d times to make it stationary"
#     q = Moving Average: "Use the last q forecast errors to improve predictions"
#
#   Example: ARIMA(2,1,2) means:
#     - Look at the last 2 values (AR=2)
#     - Difference once to remove trend (I=1)
#     - Use the last 2 errors for correction (MA=2)
#
# PERFORMANCE NOTE:
#   ARIMA is SLOW on large datasets. We mitigate this by:
#   1. Using daily aggregated data (not hourly)
#   2. Limiting training to the last 365 data points
#   3. Catching failures gracefully (sometimes ARIMA can't converge)

def arima_forecast(series, order=(2, 1, 2), steps=30):
    """
    Run ARIMA forecasting on a time series.

    Parameters:
    -----------
    series : pd.Series
        Time-indexed numeric series (daily recommended).
    order : tuple (p, d, q)
        ARIMA model order. Default (2,1,2) works well for weather data.
    steps : int
        Number of future time steps to forecast. Default 30 (days).

    Returns:
    --------
    dict: Keys → 'forecast', 'conf_int', 'mae', 'rmse', 'success'
        - forecast: pd.Series of predicted values
        - conf_int: pd.DataFrame with lower/upper confidence bounds
        - mae: Mean Absolute Error on training data
        - rmse: Root Mean Squared Error on training data
        - success: bool indicating if the model fit succeeded
    """

    # Drop NaN values
    clean = series.dropna()

    # Need a reasonable amount of data to train ARIMA
    if len(clean) < 30:
        return {
            "forecast": pd.Series(dtype=float),
            "conf_int": pd.DataFrame(),
            "mae": None,
            "rmse": None,
            "success": False,
        }

    # Limit training data to last 365 points for performance
    # ARIMA on 5 years of daily data (1825 points) can take minutes;
    # 365 points takes seconds and still captures yearly patterns.
    train = clean.tail(365)

    try:
        # Fit the ARIMA model
        # enforce_stationarity=False and enforce_invertibility=False
        # make the model more flexible and less likely to crash
        # on edge-case data
        model = ARIMA(train, order=order)
        fitted = model.fit()

        # Generate forecast for the next 'steps' time periods
        forecast_result = fitted.get_forecast(steps=steps)

        # Extract the predicted values
        forecast = forecast_result.predicted_mean

        # Extract confidence intervals (by default 95%)
        # This gives us the range: "We're 95% sure the value will be
        # between lower and upper"
        conf_int = forecast_result.conf_int()

        # Calculate accuracy metrics on the TRAINING data
        # (how well did the model fit the data it learned from?)
        train_pred = fitted.fittedvalues

        # Align the predictions with actual values
        # (fitted values might be shorter due to differencing)
        aligned_actual = train.iloc[-len(train_pred):]

        mae = round(mean_absolute_error(aligned_actual, train_pred), 2)
        rmse = round(np.sqrt(mean_squared_error(aligned_actual, train_pred)), 2)

        return {
            "forecast": forecast,
            "conf_int": conf_int,
            "mae": mae,
            "rmse": rmse,
            "success": True,
        }

    except Exception:
        # ARIMA can fail for various reasons:
        # - Data is too noisy or too constant
        # - Model can't converge within iteration limit
        # - Memory issues on very large datasets
        # We return a "failed" result instead of crashing the dashboard
        return {
            "forecast": pd.Series(dtype=float),
            "conf_int": pd.DataFrame(),
            "mae": None,
            "rmse": None,
            "success": False,
        }


# ──────────────────────────────────────────────
# FUNCTION 8: generate_insights
# ──────────────────────────────────────────────
# PURPOSE: Automatically generate human-readable text insights
#          from the data, like a weather analyst would write.
#
# WHY WE NEED IT: Charts are great, but sometimes users want
#   quick textual takeaways: "The hottest day was July 15th at 48.3°C"
#   or "Temperature shows a statistically significant upward trend."
#   This function creates those insights dynamically from the data.
#
# HOW IT WORKS:
#   We look at the data from multiple angles:
#   1. Extremes (hottest day, coldest day, rainiest day)
#   2. Trends (is the variable going up or down over time?)
#   3. Anomalies (how many unusual values did we detect?)
#   4. Variability (which months have the most variation?)
#   5. Correlations (which variables move together?)

def generate_insights(df, variable, all_variables=None):
    """
    Auto-generate text insights for a given weather variable.

    Parameters:
    -----------
    df : pd.DataFrame
        The filtered weather DataFrame (already filtered by city/date).
    variable : str
        The column to analyze (e.g., "Temperature (°C)").
    all_variables : list[str] | None
        List of all weather variable columns for correlation insights.

    Returns:
    --------
    list[str]: Human-readable insight strings.
    """

    insights = []

    # Skip if the variable doesn't exist in the DataFrame
    if variable not in df.columns:
        return ["No data available for this variable."]

    series = df[variable].dropna()

    if len(series) == 0:
        return ["No data available for analysis."]

    # ── Insight 1: Overall summary ──
    mean_val = series.mean()
    std_val = series.std()
    insights.append(
        f"Average {variable}: {mean_val:.1f} "
        f"(± {std_val:.1f} std dev) across {len(series):,} data points."
    )

    # ── Insight 2: Extremes ──
    max_val = series.max()
    min_val = series.min()
    if "time" in df.columns:
        max_idx = series.idxmax()
        min_idx = series.idxmin()
        max_date = df.loc[max_idx, "time"]
        min_date = df.loc[min_idx, "time"]

        # Format dates nicely
        if hasattr(max_date, "strftime"):
            insights.append(
                f"Highest recorded: {max_val:.1f} on "
                f"{max_date.strftime('%B %d, %Y')}."
            )
            insights.append(
                f"Lowest recorded: {min_val:.1f} on "
                f"{min_date.strftime('%B %d, %Y')}."
            )
        else:
            insights.append(f"Range: {min_val:.1f} to {max_val:.1f}.")
    else:
        insights.append(f"Range: {min_val:.1f} to {max_val:.1f}.")

    # ── Insight 3: Trend direction ──
    if len(series) >= 10:
        trend = linear_trend(series)
        slope = trend["slope"]
        r_sq = trend["r_squared"]
        p_val = trend["p_value"]

        if p_val < 0.05:
            direction = "upward" if slope > 0 else "downward"
            insights.append(
                f"Statistically significant {direction} trend detected "
                f"(slope: {slope:.4f} per time step, R²={r_sq:.3f}, "
                f"p={p_val:.4f})."
            )
        else:
            insights.append(
                "No statistically significant trend detected in the data."
            )

    # ── Insight 4: Anomaly count ──
    anomalies = detect_anomalies_zscore(series, threshold=2)
    anomaly_count = anomalies.sum()
    anomaly_pct = (anomaly_count / len(series)) * 100
    insights.append(
        f"Anomalies detected: {anomaly_count:,} points "
        f"({anomaly_pct:.1f}% of data) exceed ±2 standard deviations."
    )

    # ── Insight 5: Seasonal patterns ──
    if "Season" in df.columns:
        seasonal_means = df.groupby("Season")[variable].mean()
        highest_season = seasonal_means.idxmax()
        lowest_season = seasonal_means.idxmin()
        insights.append(
            f"Seasonal pattern: highest average in {highest_season} "
            f"({seasonal_means[highest_season]:.1f}), "
            f"lowest in {lowest_season} ({seasonal_means[lowest_season]:.1f})."
        )

    # ── Insight 6: Monthly variability ──
    if "Month" in df.columns:
        monthly_std = df.groupby("Month")[variable].std()
        most_variable_month = monthly_std.idxmax()
        month_names = {
            1: "January", 2: "February", 3: "March", 4: "April",
            5: "May", 6: "June", 7: "July", 8: "August",
            9: "September", 10: "October", 11: "November", 12: "December",
        }
        insights.append(
            f"Most variable month: {month_names.get(most_variable_month, most_variable_month)} "
            f"(std dev: {monthly_std[most_variable_month]:.1f})."
        )

    # ── Insight 7: Correlation highlights ──
    if all_variables and len(all_variables) > 1:
        other_vars = [v for v in all_variables if v in df.columns and v != variable]
        if other_vars:
            correlations = {}
            for other in other_vars:
                if df[other].dropna().nunique() > 1:
                    corr = df[[variable, other]].dropna().corr().iloc[0, 1]
                    correlations[other] = corr

            if correlations:
                # Find strongest correlation (positive or negative)
                strongest = max(correlations, key=lambda k: abs(correlations[k]))
                corr_val = correlations[strongest]
                strength = (
                    "strong" if abs(corr_val) > 0.7
                    else "moderate" if abs(corr_val) > 0.4
                    else "weak"
                )
                direction = "positive" if corr_val > 0 else "negative"
                insights.append(
                    f"Strongest correlation: {strength} {direction} "
                    f"relationship with {strongest} (r={corr_val:.3f})."
                )

    return insights
