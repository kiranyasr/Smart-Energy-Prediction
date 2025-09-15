import pandas as pd

def clean_dataset(df):
    # Example cleaning: drop NA, reset index
    df = df.dropna().reset_index(drop=True)
    return df
