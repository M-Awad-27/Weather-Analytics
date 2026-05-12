"""
visualizer.py — Plotly Visualization Module
=============================================
This module contains ALL chart-building functions for the dashboard.
Every function returns a plotly.graph_objects.Figure with a consistent
dark theme that matches our Streamlit config.

What this file does:
1.  Time series line charts (with rolling averages, anomalies, trend)
2.  Histograms with KDE overlays
3.  Monthly box plots
4.  Seasonal violin plots
5.  Correlation heatmaps (annotated)
6.  Scatter plots with OLS regression lines
7.  Grouped bar charts (month × year)
8.  GitHub-style calendar heatmaps
9.  Forecast charts (historical + predicted + confidence intervals)
10. Seasonal decomposition subplots
"""

# ──────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ──────────────────────────────────────────────
# THEME CONSTANTS
# ──────────────────────────────────────────────
# We define all colours and layout defaults here so every chart
# looks consistent.  Changing one value here updates ALL charts.

# Primary accent palette — harmonious purples / teals / warm tones
ACCENT     = "#6C63FF"   # Primary purple (matches Streamlit theme)
ACCENT2    = "#00D4AA"   # Teal / mint green
ACCENT3    = "#FF6B6B"   # Warm coral (anomalies / warnings)
ACCENT4    = "#FFD93D"   # Gold / amber
ACCENT5    = "#4ECDC4"   # Soft cyan

# Gradient-friendly series colours (used for multi-line / multi-bar charts)
SERIES_COLORS = [
    "#6C63FF", "#00D4AA", "#FF6B6B", "#FFD93D",
    "#4ECDC4", "#FF8C42", "#A78BFA", "#38BDF8",
    "#F472B6", "#34D399", "#FBBF24", "#F87171",
]

# Background / surface colours (match .streamlit/config.toml)
BG_COLOR   = "#0E1117"
CARD_BG    = "#1A1D29"
GRID_COLOR = "rgba(255,255,255,0.06)"
TEXT_COLOR  = "#FAFAFA"
MUTED_TEXT  = "rgba(255,255,255,0.5)"

# Shared layout dict applied to every figure via fig.update_layout()
BASE_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor=BG_COLOR,
    plot_bgcolor=CARD_BG,
    font=dict(family="Inter, sans-serif", color=TEXT_COLOR, size=13),
    margin=dict(l=60, r=30, t=50, b=50),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor="rgba(255,255,255,0.1)",
        borderwidth=1,
        font=dict(size=11),
    ),
    xaxis=dict(gridcolor=GRID_COLOR, zeroline=False),
    yaxis=dict(gridcolor=GRID_COLOR, zeroline=False),
)

# Month labels for axis formatting
MONTH_LABELS = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
    5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
    9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}


def _apply_base(fig, title="", height=500):
    """Apply the shared dark-theme layout to any figure."""
    fig.update_layout(**BASE_LAYOUT, title=dict(text=title, x=0.5, font=dict(size=16)), height=height)
    return fig


# ──────────────────────────────────────────────
# 1. TIME SERIES — line + rolling + anomalies + trend
# ──────────────────────────────────────────────

def plot_time_series(
    df, column, rolling_data=None, anomalies=None, trend_data=None
):
    """
    Line chart of a weather variable over time, optionally overlaid with
    rolling averages, anomaly markers, and a linear trend line.

    Parameters
    ----------
    df : pd.DataFrame        Must contain 'time' and *column*.
    column : str              The weather variable to plot.
    rolling_data : dict|None  Output of analysis.rolling_statistics().
    anomalies : pd.Series|None  Boolean mask from detect_anomalies_zscore().
    trend_data : dict|None    Output of analysis.linear_trend().

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()

    # Main line
    fig.add_trace(go.Scatter(
        x=df["time"], y=df[column],
        mode="lines", name=column,
        line=dict(color=ACCENT, width=1.5),
        opacity=0.85,
    ))

    # Rolling averages (7-day and 30-day)
    if rolling_data:
        colors = {7: ACCENT2, 30: ACCENT4}
        for window, data in rolling_data.items():
            fig.add_trace(go.Scatter(
                x=df["time"], y=data["mean"],
                mode="lines", name=f"{window}-Day Avg",
                line=dict(color=colors.get(window, ACCENT5), width=2, dash="dot"),
            ))

    # Trend line
    if trend_data and "trend_line" in trend_data:
        fig.add_trace(go.Scatter(
            x=df["time"], y=trend_data["trend_line"],
            mode="lines", name="Trend",
            line=dict(color=ACCENT4, width=2, dash="dash"),
            opacity=0.7,
        ))

    # Anomaly dots
    if anomalies is not None and anomalies.any():
        anom_df = df[anomalies]
        fig.add_trace(go.Scatter(
            x=anom_df["time"], y=anom_df[column],
            mode="markers", name="Anomaly",
            marker=dict(color=ACCENT3, size=7, symbol="diamond",
                        line=dict(width=1, color="white")),
        ))

    return _apply_base(fig, f"{column} — Time Series", height=480)


# ──────────────────────────────────────────────
# 2. HISTOGRAM + KDE
# ──────────────────────────────────────────────

def plot_histogram_kde(series, column_name):
    """
    Histogram with a smooth KDE (kernel density estimate) overlay.

    Parameters
    ----------
    series : pd.Series   Numeric values to plot.
    column_name : str     Label for the x-axis.

    Returns
    -------
    go.Figure
    """
    clean = series.dropna()

    fig = go.Figure()

    # Histogram bars
    fig.add_trace(go.Histogram(
        x=clean, nbinsx=50, name="Frequency",
        marker=dict(color=ACCENT, line=dict(width=0.5, color=CARD_BG)),
        opacity=0.75,
    ))

    # KDE curve (approximate via numpy histogram + smoothing)
    counts, bin_edges = np.histogram(clean, bins=80, density=True)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
    # Simple moving-average smoothing for the KDE line
    kernel_size = max(3, len(counts) // 10)
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.convolve(counts, kernel, mode="same")

    fig.add_trace(go.Scatter(
        x=bin_centres, y=smoothed * len(clean) * (bin_edges[1] - bin_edges[0]),
        mode="lines", name="KDE",
        line=dict(color=ACCENT2, width=2.5),
        yaxis="y",
    ))

    fig.update_layout(
        xaxis_title=column_name,
        yaxis_title="Frequency",
        barmode="overlay",
    )
    return _apply_base(fig, f"{column_name} — Distribution", height=440)


# ──────────────────────────────────────────────
# 3. MONTHLY BOX PLOT
# ──────────────────────────────────────────────

def plot_boxplot_monthly(df, column):
    """
    Box plot grouped by calendar month (1–12).

    Parameters
    ----------
    df : pd.DataFrame   Must contain 'Month' and *column*.
    column : str         The weather variable.

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()

    for month in range(1, 13):
        month_data = df[df["Month"] == month][column].dropna()
        fig.add_trace(go.Box(
            y=month_data,
            name=MONTH_LABELS[month],
            marker_color=SERIES_COLORS[month - 1],
            boxmean="sd",           # show mean + std marker
            line=dict(width=1.2),
        ))

    fig.update_layout(
        xaxis_title="Month",
        yaxis_title=column,
        showlegend=False,
    )
    return _apply_base(fig, f"{column} — Monthly Box Plot", height=460)


# ──────────────────────────────────────────────
# 4. SEASONAL VIOLIN PLOT
# ──────────────────────────────────────────────

def plot_violin_seasonal(df, column):
    """
    Violin plot grouped by season (Winter / Spring / Summer / Autumn).

    Parameters
    ----------
    df : pd.DataFrame   Must contain 'Season' and *column*.
    column : str         The weather variable.

    Returns
    -------
    go.Figure
    """
    season_order = ["Winter", "Spring", "Summer", "Autumn"]
    season_colors = {
        "Winter": "#38BDF8", "Spring": "#34D399",
        "Summer": "#FBBF24", "Autumn": "#F87171",
    }

    fig = go.Figure()

    for season in season_order:
        season_data = df[df["Season"] == season][column].dropna()
        if len(season_data) == 0:
            continue
        fig.add_trace(go.Violin(
            y=season_data, name=season,
            box_visible=True, meanline_visible=True,
            fillcolor=season_colors.get(season, ACCENT),
            line_color="white", opacity=0.7,
        ))

    fig.update_layout(
        xaxis_title="Season",
        yaxis_title=column,
        showlegend=False,
    )
    return _apply_base(fig, f"{column} — Seasonal Distribution", height=460)


# ──────────────────────────────────────────────
# 5. CORRELATION HEATMAP
# ──────────────────────────────────────────────

def plot_correlation_heatmap(corr_matrix):
    """
    Annotated heatmap from a correlation matrix DataFrame.

    Parameters
    ----------
    corr_matrix : pd.DataFrame   Square correlation matrix (–1 to +1).

    Returns
    -------
    go.Figure
    """
    labels = corr_matrix.columns.tolist()
    z = corr_matrix.values

    # Shorten long column names for readability
    short = [l.split(" (")[0] if " (" in l else l for l in labels]

    fig = go.Figure(data=go.Heatmap(
        z=z, x=short, y=short,
        colorscale=[
            [0.0, "#0D47A1"],    # strong negative → deep blue
            [0.25, "#42A5F5"],
            [0.5, "#1A1D29"],    # zero → dark background
            [0.75, "#EF5350"],
            [1.0, "#B71C1C"],    # strong positive → deep red
        ],
        zmid=0,
        text=np.round(z, 2),
        texttemplate="%{text}",
        textfont=dict(size=11, color="white"),
        hovertemplate="(%{x}, %{y}): %{z:.3f}<extra></extra>",
        colorbar=dict(title="r", thickness=12),
    ))

    fig.update_layout(xaxis=dict(side="bottom"), yaxis=dict(autorange="reversed"))
    return _apply_base(fig, "Correlation Heatmap", height=500)


# ──────────────────────────────────────────────
# 6. SCATTER + REGRESSION
# ──────────────────────────────────────────────

def plot_scatter_regression(df, x_col, y_col):
    """
    Scatter plot of two variables with an OLS regression line.

    Parameters
    ----------
    df : pd.DataFrame   Must contain *x_col* and *y_col*.
    x_col, y_col : str  Column names.

    Returns
    -------
    go.Figure
    """
    from scipy import stats as sp_stats

    clean = df[[x_col, y_col]].dropna()
    x = clean[x_col].values
    y = clean[y_col].values

    fig = go.Figure()

    # Scatter points
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers", name="Data",
        marker=dict(color=ACCENT, size=4, opacity=0.4,
                    line=dict(width=0)),
    ))

    # Regression line
    if len(x) >= 2:
        slope, intercept, rvalue, pvalue, _ = sp_stats.linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept
        fig.add_trace(go.Scatter(
            x=x_line, y=y_line, mode="lines",
            name=f"OLS (R²={rvalue**2:.3f})",
            line=dict(color=ACCENT3, width=2.5, dash="dash"),
        ))

    fig.update_layout(xaxis_title=x_col, yaxis_title=y_col)
    return _apply_base(fig, f"{x_col}  vs  {y_col}", height=460)


# ──────────────────────────────────────────────
# 7. GROUPED BAR — month × year
# ──────────────────────────────────────────────

def plot_monthly_bar_by_year(df, column):
    """
    Grouped bar chart showing monthly averages coloured by year.

    Parameters
    ----------
    df : pd.DataFrame   Must contain 'Month', 'Year', and *column*.
    column : str         The weather variable.

    Returns
    -------
    go.Figure
    """
    pivot = df.groupby(["Year", "Month"])[column].mean().reset_index()
    years = sorted(pivot["Year"].unique())

    fig = go.Figure()
    for i, year in enumerate(years):
        yr_data = pivot[pivot["Year"] == year]
        fig.add_trace(go.Bar(
            x=[MONTH_LABELS.get(m, m) for m in yr_data["Month"]],
            y=yr_data[column],
            name=str(year),
            marker_color=SERIES_COLORS[i % len(SERIES_COLORS)],
            opacity=0.85,
        ))

    fig.update_layout(
        barmode="group",
        xaxis_title="Month",
        yaxis_title=f"Avg {column}",
    )
    return _apply_base(fig, f"{column} — Monthly Avg by Year", height=460)


# ──────────────────────────────────────────────
# 8. CALENDAR HEATMAP (GitHub-style)
# ──────────────────────────────────────────────

def plot_calendar_heatmap(df, column):
    """
    GitHub contribution-style heatmap showing daily values across weeks.

    Parameters
    ----------
    df : pd.DataFrame   Must contain 'time' and *column*.
    column : str         The weather variable.

    Returns
    -------
    go.Figure
    """
    if "time" not in df.columns:
        return _apply_base(go.Figure(), "Calendar Heatmap — no time column")

    daily = df.groupby(df["time"].dt.date)[column].mean().reset_index()
    daily.columns = ["date", "value"]
    daily["date"] = pd.to_datetime(daily["date"])
    daily["week"] = daily["date"].dt.isocalendar().week.astype(int)
    daily["year"] = daily["date"].dt.year
    daily["dow"] = daily["date"].dt.dayofweek  # 0=Mon … 6=Sun

    years = sorted(daily["year"].unique())
    n_years = len(years)

    fig = make_subplots(
        rows=n_years, cols=1,
        subplot_titles=[str(y) for y in years],
        vertical_spacing=0.06,
    )

    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    for idx, year in enumerate(years, 1):
        yr = daily[daily["year"] == year]
        fig.add_trace(go.Heatmap(
            x=yr["week"], y=yr["dow"],
            z=yr["value"],
            colorscale=[
                [0, "#161B22"],
                [0.25, "#0E4429"],
                [0.5, "#006D32"],
                [0.75, "#26A641"],
                [1, "#39D353"],
            ],
            showscale=(idx == 1),
            hovertemplate="Week %{x}, %{text}<br>%{z:.1f}<extra></extra>",
            text=[d.strftime("%b %d") for d in yr["date"]],
            colorbar=dict(title=column, thickness=10) if idx == 1 else None,
        ), row=idx, col=1)

        fig.update_yaxes(
            tickvals=list(range(7)), ticktext=day_labels,
            row=idx, col=1,
        )
        fig.update_xaxes(title_text="Week of Year" if idx == n_years else "", row=idx, col=1)

    height = max(300, 180 * n_years)
    return _apply_base(fig, f"{column} — Calendar Heatmap", height=height)


# ──────────────────────────────────────────────
# 9. FORECAST CHART
# ──────────────────────────────────────────────

def plot_forecast(historical_series, forecast_result):
    """
    Historical line + dashed forecast line + confidence interval shading.

    Parameters
    ----------
    historical_series : pd.Series   The observed values (time-indexed).
    forecast_result : dict           Output of analysis.arima_forecast().

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()

    # Historical data (last 90 points for context)
    hist = historical_series.dropna().tail(90)
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist.values,
        mode="lines", name="Historical",
        line=dict(color=ACCENT, width=2),
    ))

    if not forecast_result.get("success"):
        fig.add_annotation(
            text="ARIMA forecast could not converge for this data.",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14, color=ACCENT3),
        )
        return _apply_base(fig, "Forecast (ARIMA)", height=460)

    forecast = forecast_result["forecast"]
    conf_int = forecast_result["conf_int"]

    # Forecast line
    fig.add_trace(go.Scatter(
        x=forecast.index, y=forecast.values,
        mode="lines", name="Forecast",
        line=dict(color=ACCENT4, width=2.5, dash="dash"),
    ))

    # Confidence interval shading
    if not conf_int.empty:
        lower = conf_int.iloc[:, 0]
        upper = conf_int.iloc[:, 1]
        fig.add_trace(go.Scatter(
            x=list(upper.index) + list(lower.index[::-1]),
            y=list(upper.values) + list(lower.values[::-1]),
            fill="toself", fillcolor="rgba(255,217,61,0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name="95% Confidence",
            hoverinfo="skip",
        ))

    # Accuracy annotation
    mae = forecast_result.get("mae")
    rmse = forecast_result.get("rmse")
    if mae is not None:
        fig.add_annotation(
            text=f"MAE: {mae}  |  RMSE: {rmse}",
            xref="paper", yref="paper", x=0.02, y=0.98,
            showarrow=False,
            font=dict(size=12, color=ACCENT2),
            bgcolor="rgba(0,0,0,0.5)", borderpad=4,
        )

    return _apply_base(fig, "Forecast (ARIMA)", height=480)


# ──────────────────────────────────────────────
# 10. SEASONAL DECOMPOSITION SUBPLOTS
# ──────────────────────────────────────────────

def plot_decomposition(decompose_result):
    """
    Four vertically stacked subplots: Observed, Trend, Seasonal, Residual.

    Parameters
    ----------
    decompose_result : DecomposeResult
        Output of analysis.seasonal_decompose_series().

    Returns
    -------
    go.Figure
    """
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        subplot_titles=["Observed", "Trend", "Seasonal", "Residual"],
        vertical_spacing=0.06,
    )

    components = [
        (decompose_result.observed, ACCENT,  1),
        (decompose_result.trend,    ACCENT2, 2),
        (decompose_result.seasonal, ACCENT4, 3),
        (decompose_result.resid,    ACCENT3, 4),
    ]

    for series, color, row in components:
        fig.add_trace(go.Scatter(
            x=series.index, y=series.values,
            mode="lines", line=dict(color=color, width=1.5),
            showlegend=False,
        ), row=row, col=1)

    # Apply grid styling to all subplots
    for i in range(1, 5):
        fig.update_yaxes(gridcolor=GRID_COLOR, row=i, col=1)
        fig.update_xaxes(gridcolor=GRID_COLOR, row=i, col=1)

    return _apply_base(fig, "Seasonal Decomposition", height=700)
