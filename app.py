"""
Global Weather Dashboard
========================
Main Streamlit Application. Ties together loader, analysis, and visualizer.
"""

import streamlit as st
import pandas as pd
from datetime import datetime

# Import our custom modules
from utils import loader, analysis, visualizer

# ──────────────────────────────────────────────
# 1. PAGE CONFIG & CUSTOM CSS
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="Global Weather Analytics",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium dark theme / glassmorphism styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif !important;
    }

    /* Animations */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulseGlow {
        0% { box-shadow: 0 4px 6px rgba(0,0,0,0.3), 0 0 5px rgba(108, 99, 255, 0.2); }
        50% { box-shadow: 0 4px 6px rgba(0,0,0,0.3), 0 0 20px rgba(108, 99, 255, 0.6); }
        100% { box-shadow: 0 4px 6px rgba(0,0,0,0.3), 0 0 5px rgba(108, 99, 255, 0.2); }
    }

    /* Main Container Fade In */
    .main .block-container {
        animation: fadeInUp 0.8s ease-out forwards;
    }

    /* Glassmorphism Metric Cards */
    div[data-testid="metric-container"] {
        background: rgba(26, 29, 41, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        animation: fadeInUp 0.5s ease-out forwards;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-8px) scale(1.02);
        border-color: #6C63FF;
        animation: pulseGlow 2s infinite;
    }

    /* Hide the Streamlit top menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Premium Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: transparent;
        padding-bottom: 5px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(26, 29, 41, 0.5);
        border-radius: 8px;
        padding: 10px 24px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        color: #A0AEC0;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #FAFAFA;
        background-color: rgba(108, 99, 255, 0.2);
        border-color: rgba(108, 99, 255, 0.4);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6C63FF 0%, #483D8B 100%) !important;
        border-color: #6C63FF !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(108, 99, 255, 0.4);
    }
    
    /* Hyperlink styling (specifically for the Made By link) */
    a {
        color: #00D4AA !important;
        text-decoration: none;
        position: relative;
        font-weight: 600;
        transition: color 0.3s ease;
    }
    a:hover {
        color: #6C63FF !important;
        text-shadow: 0 0 8px rgba(108, 99, 255, 0.6);
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# 2. SIDEBAR (FILTERS & CONTROLS)
# ──────────────────────────────────────────────

st.sidebar.title("🌍 Global Weather")
st.sidebar.markdown("Made by [Muhammad Awad](https://github.com/M-Awad-27)")
st.sidebar.markdown("Advanced analytics dashboard for 7 international cities.")

st.sidebar.header("1. Location & Time")

# City Selection
cities_list = list(loader.CITIES.keys())
selected_city = st.sidebar.selectbox("Select City", cities_list, index=0)

# We want roughly a 5-year window by default for good analysis, ending today
default_end = datetime.utcnow().date()
default_start = default_end.replace(year=default_end.year - 5)

# Date Range Picker
# (Note: For the free tier API, fetching too much data at once takes time. 
# 5 years per city is a good sweet spot).
date_range = st.sidebar.date_input(
    "Date Range",
    value=(default_start, default_end),
    max_value=default_end
)

st.sidebar.header("2. Analysis Parameters")

# Target Variable Selection
weather_vars = list(loader.COLUMN_RENAME_MAP.values())
target_variable = st.sidebar.selectbox("Target Variable", weather_vars, index=0)

# Data Aggregation
agg_options = {"Daily": "D", "Weekly": "W", "Monthly": "ME"}
agg_choice = st.sidebar.radio("Data Aggregation", list(agg_options.keys()), index=0)
freq = agg_options[agg_choice]


# ──────────────────────────────────────────────
# 3. DATA LOADING & PROCESSING
# ──────────────────────────────────────────────

# Ensure the user has selected a valid start and end date
if len(date_range) != 2:
    st.warning("Please select a complete date range (Start and End).")
    st.stop()

start_date, end_date = date_range

# Show a spinner while loading (will be instant if cached)
with st.spinner("Fetching and processing data..."):
    # Load raw data
    raw_df = loader.load_data(
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d")
    )
    
    # Filter to selected city
    city_df = raw_df[raw_df["City"] == selected_city].copy()
    
    # Aggregate data based on user selection (Daily/Weekly/Monthly)
    if freq != "H": # 'H' would be hourly, but we only offer D/W/ME
        agg_df = loader.aggregate_data(city_df, freq=freq)
    else:
        agg_df = city_df

# If no data returned (e.g., API failure or dates in future), stop gracefully
if agg_df.empty:
    st.error("No data available for the selected filters.")
    st.stop()


# ──────────────────────────────────────────────
# 4. TOP METRIC CARDS
# ──────────────────────────────────────────────

st.title(f"Weather Analytics: {selected_city}")
st.markdown(f"**{start_date.strftime('%b %d, %Y')}** to **{end_date.strftime('%b %d, %Y')}** ({agg_choice} Aggregation)")

# Get statistical summary for the selected variable
stats = analysis.compute_summary_stats(agg_df[target_variable])

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Average", f"{stats['mean']:.1f}")
col2.metric("Maximum", f"{stats['max']:.1f}")
col3.metric("Minimum", f"{stats['min']:.1f}")
col4.metric("Std Dev", f"{stats['std']:.1f}")
col5.metric("Data Points", f"{stats['count']:,}")

st.markdown("---")


# ──────────────────────────────────────────────
# 5. TABS INTERFACE
# ──────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Time Series", 
    "📊 Distribution", 
    "🔗 Correlation", 
    "📅 Calendar", 
    "🔮 Forecast"
])

# ─── TAB 1: TIME SERIES ───
with tab1:
    st.subheader(f"Historical Trend: {target_variable}")
    
    # Calculate overlays for the chart
    rolling = analysis.rolling_statistics(agg_df[target_variable], windows=[7, 30])
    anomalies = analysis.detect_anomalies_zscore(agg_df[target_variable], threshold=2.5)
    trend = analysis.linear_trend(agg_df[target_variable])
    
    # Plot main time series
    fig_ts = visualizer.plot_time_series(
        agg_df, target_variable, 
        rolling_data=rolling, 
        anomalies=anomalies, 
        trend_data=trend
    )
    st.plotly_chart(fig_ts, use_container_width=True)
    
    # Add an expander for deep-dive decomposition
    with st.expander("🔍 Deep Dive: Seasonal Decomposition"):
        st.markdown("Breaks down the time series into underlying Trend, Seasonality, and random Noise.")
        # Re-set index for decomposition
        decomp_series = agg_df.set_index("time")[target_variable]
        decomp_result = analysis.seasonal_decompose_series(decomp_series)
        
        if decomp_result:
            fig_decomp = visualizer.plot_decomposition(decomp_result)
            st.plotly_chart(fig_decomp, use_container_width=True)
        else:
            st.info("Not enough data points or cycles to perform seasonal decomposition.")

# ─── TAB 2: DISTRIBUTION ───
with tab2:
    st.subheader("Data Distribution & Spread")
    col2_1, col2_2 = st.columns(2)
    
    with col2_1:
        # Histogram + KDE
        fig_hist = visualizer.plot_histogram_kde(agg_df[target_variable], target_variable)
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with col2_2:
        # Monthly Box Plots
        if "Month" in agg_df.columns:
            fig_box = visualizer.plot_boxplot_monthly(agg_df, target_variable)
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.warning("Monthly breakdown requires daily or weekly aggregation.")
            
    # Seasonal Violin Plots
    if "Season" in agg_df.columns:
        fig_violin = visualizer.plot_violin_seasonal(agg_df, target_variable)
        st.plotly_chart(fig_violin, use_container_width=True)

# ─── TAB 3: CORRELATION ───
with tab3:
    st.subheader("Variable Relationships")
    
    # Year-over-year monthly trends
    if "Year" in agg_df.columns and "Month" in agg_df.columns:
        st.markdown("### Year-Over-Year Monthly Averages")
        fig_bar = visualizer.plot_monthly_bar_by_year(agg_df, target_variable)
        st.plotly_chart(fig_bar, use_container_width=True)

# ─── TAB 4: CALENDAR HEATMAP ───
with tab4:
    st.subheader("Daily Intensity Heatmap")
    st.markdown("Visualizes daily patterns across the year, similar to GitHub contribution charts.")
    
    if agg_choice == "Daily":
        fig_cal = visualizer.plot_calendar_heatmap(agg_df, target_variable)
        st.plotly_chart(fig_cal, use_container_width=True)
    else:
        st.info("Calendar heatmap is only available when Data Aggregation is set to 'Daily'.")

# ─── TAB 5: FORECAST ───
with tab5:
    st.subheader(f"Predictive Forecast: {target_variable}")
    st.markdown("Using ARIMA (AutoRegressive Integrated Moving Average) modeling to predict the next 30 steps based on historical patterns.")
    
    if agg_choice == "Daily":
        with st.spinner("Training ARIMA model... (this may take a few seconds)"):
            # Set index to time for time-series forecasting
            ts_data = agg_df.set_index("time")[target_variable]
            
            # Predict 30 days out
            forecast_result = analysis.arima_forecast(ts_data, steps=30)
            
            fig_forecast = visualizer.plot_forecast(ts_data, forecast_result)
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            if forecast_result.get("success"):
                st.success("Forecast generated successfully.")
    else:
        st.info("Forecasting is currently optimized for 'Daily' aggregated data. Please switch aggregation to Daily in the sidebar.")


# ──────────────────────────────────────────────
# 6. AUTO-GENERATED INSIGHTS (SIDEBAR)
# ──────────────────────────────────────────────

st.sidebar.markdown("---")
st.sidebar.header("💡 AI Insights")

# Generate automatic text insights based on the filtered data
insights = analysis.generate_insights(agg_df, target_variable, all_variables=weather_vars)

for text in insights:
    st.sidebar.info(text)

# Also show data quality summary
st.sidebar.markdown("---")
with st.sidebar.expander("🛠️ Data Quality Summary"):
    quality = loader.get_data_quality_summary(agg_df)
    st.write(f"**Total Rows:** {quality['total_rows']:,}")
    
    st.write("**Missing Values:**")
    st.json(quality['missing_pct'])
    
    st.write("**Outliers Detected:**")
    st.json(quality['outlier_counts'])
