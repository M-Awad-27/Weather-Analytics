# 🌍 Global Weather Analytics Dashboard

A premium, interactive Streamlit application for analyzing and forecasting historical weather data for 7 major international cities. 

This dashboard connects to the **Open-Meteo Archive API** to fetch 5 years of historical hourly weather data. It cleans, processes, and aggregates the data to perform advanced statistical analysis and time-series forecasting, all wrapped in a sleek, glassmorphism-styled UI.

![Dashboard Preview](https://img.shields.io/badge/UI-Dark_Mode-6C63FF?style=flat-square) ![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python) ![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=flat-square&logo=streamlit) ![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75?style=flat-square&logo=plotly)

### 🔴 **[Live Demo: Check out the deployed application here!](https://weather-analytics-app1.streamlit.app/)**

## ✨ Key Features

- **Automated Data Pipeline**: Dynamically fetches data from the Open-Meteo API (no API keys required) and caches it locally as a CSV for instant reloading.
- **Dynamic Filtering**: Easily switch between 7 global cities (Karachi, Mumbai, Riyadh, Dubai, New York, Auckland, Sydney), select specific date ranges, and toggle data aggregation between Daily, Weekly, and Monthly.
- **AI-Generated Insights**: An automated text generation engine that translates complex metrics (like statistical significance, strongest correlations, and anomaly counts) into easily digestible human-readable summaries.
- **5 Advanced Analysis Tabs**:
  1. **Time Series**: Interactive line charts with 7-day/30-day rolling averages, red-dot anomaly markers, and OLS trend lines. Includes a deep-dive Seasonal Decomposition module.
  2. **Distribution**: Histograms with KDE (Kernel Density Estimate) curves, Monthly Box Plots, and Seasonal Violin Plots.
  3. **Correlation**: Annotated Heatmaps, Scatter Plots with regression lines, and Year-over-Year grouped bar charts.
  4. **Calendar Heatmap**: A GitHub-contribution style matrix visualizing daily intensity across 52 weeks.
  5. **Forecast**: A 30-day predictive forecast utilizing an ARIMA model, complete with 95% confidence interval shading.

## 🚀 Setup & Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/M-Awad-27/Weather-Analytics.git
   cd Weather-Analytics
   ```
2. Create and activate a virtual environment (optional but recommended).
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
5. Navigate to `http://localhost:8501` in your browser. *(Note: The initial run may take ~30 seconds as it downloads 5 years of hourly data from the API).*

---

## 🧮 Statistical Formulas & Methods Used

This dashboard doesn't just plot raw data; it applies robust statistical methods to extract meaningful patterns. Here are the core formulas powering the backend (`utils/analysis.py`):

### 1. Z-Score Anomaly Detection
Identifies extreme weather events (e.g., freak heatwaves or storms). We flag data points that fall more than 2.5 standard deviations from the mean.
> **Formula:**  
> $$Z = \frac{X - \mu}{\sigma}$$
> *(Where $X$ is the value, $\mu$ is the mean, and $\sigma$ is the standard deviation. Anomalies are where $|Z| > 2.5$)*

### 2. Linear Regression (Trend Lines)
Calculates if a weather variable (like Temperature) is trending upwards or downwards over time using the Ordinary Least Squares (OLS) method.
> **Formula:**  
> $$Y = mX + b$$
> *(Where $m$ is the slope/trend, and $b$ is the intercept. We also compute the $R^2$ value to determine how well the line fits the data).*

### 3. Pearson Correlation Coefficient
Measures the linear relationship between two different weather variables (e.g., Humidity vs. Apparent Temperature). 
> **Formula:**  
> $$r = \frac{\sum (X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum (X_i - \bar{X})^2 \sum (Y_i - \bar{Y})^2}}$$
> *(Values range from -1.0 to 1.0)*

### 4. Seasonal Decomposition (STL)
Breaks down a time-series dataset into three distinct components to isolate underlying patterns from random noise.
> **Formula (Additive Model):**  
> $$Y_t = T_t + S_t + R_t$$
> *(Where $Y_t$ is the observed data, $T_t$ is the Trend, $S_t$ is the Seasonality, and $R_t$ is the Residual/Noise)*

### 5. ARIMA Forecasting
AutoRegressive Integrated Moving Average (ARIMA) is used in the Forecast tab to predict future weather patterns based on historical lags and errors. We default to an `ARIMA(2,1,2)` model.
> **Components:**  
> - **AR (p=2)**: Uses the past 2 values to predict the next.  
> - **I (d=1)**: Differences the data once to make it stationary (removing trends).  
> - **MA (q=2)**: Uses the past 2 forecast errors to correct the prediction.  

---

## 📂 Project Structure
```text
weather_dashboard/
├── app.py                    # Main Streamlit application & UI layout
├── requirements.txt          # Python dependencies
├── .gitignore                
├── data/                     # Ignored directory where cached CSVs are saved
└── utils/                    
    ├── loader.py             # API fetching, data cleaning, aggregation, & IQR outlier detection
    ├── analysis.py           # Statistical math, ARIMA, regression, & AI text generation
    └── visualizer.py         # Plotly figure generation and dark-theme formatting
```

## 👨‍💻 Author
Made by **[Muhammad Awad](https://github.com/M-Awad-27)**.
