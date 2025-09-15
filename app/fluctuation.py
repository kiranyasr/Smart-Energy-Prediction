import pandas as pd

def calculate_fluctuations(df):
    # Expect columns: "Solar", "Wind"
    
    def process(series, label):
        abs_fluct = series.diff().fillna(0)
        pct_fluct = series.pct_change().fillna(0) * 100
        rolling_std = series.rolling(window=7).std().fillna(0)
        rolling_var = series.rolling(window=7).var().fillna(0)
        volatility_index = abs_fluct.std()
        
        return pd.DataFrame({
            "Date": df.index,
            f"{label}_Value": series,
            f"{label}_AbsFluct": abs_fluct,
            f"{label}_PctFluct": pct_fluct,
            f"{label}_RollingStd": rolling_std,
            f"{label}_RollingVar": rolling_var,
            f"{label}_VolatilityIndex": [volatility_index]*len(series)
        })
    
    solar = process(df["Solar"], "Solar")
    wind = process(df["Wind"], "Wind")
    
    # Combined (average of Solar & Wind)
    combined_series = (df["Solar"] + df["Wind"]) / 2
    combined = process(combined_series, "Combined")
    
    return solar, wind, combined
