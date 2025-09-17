from flask import Flask, render_template, request, session, redirect, url_for, send_file
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_secret_key_here' 
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Day 1: Data Cleaning ---
def clean_data(df):
    df.drop_duplicates(inplace=True)
    df['Date and Hour'] = pd.to_datetime(df['Date and Hour'], utc=True, errors='coerce')
    df.dropna(subset=['Date and Hour'], inplace=True)
    scaler = MinMaxScaler()
    df[['Production']] = scaler.fit_transform(df[['Production']])
    return df

@app.route('/', methods=['GET', 'POST'])
def upload_page():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            return render_template('upload.html', error='No file selected.')
        
        if file and file.filename.endswith('.csv'):
            original_filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(original_filepath)
            
            df_preview = pd.read_csv(original_filepath)
            preview_html = df_preview.head(10).to_html(classes='data-table', index=False)
            
            cleaned_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'cleaned_data.csv')
            df_to_clean = pd.read_csv(original_filepath)
            cleaned_df = clean_data(df_to_clean)
            cleaned_df.to_csv(cleaned_filepath, index=False)
            
            session['cleaned_file_path'] = cleaned_filepath
            return render_template('upload.html', success=True, filename=file.filename, preview=preview_html)

    return render_template('upload.html')

# --- Day 2: Fluctuation Analysis ---
@app.route('/dashboard')
def dashboard():
    cleaned_file = session.get('cleaned_file_path')
    if not cleaned_file or not os.path.exists(cleaned_file):
        return redirect(url_for('upload_page'))

    df = pd.read_csv(cleaned_file, parse_dates=['Date and Hour'], index_col='Date and Hour')
    df = df[~df.index.duplicated(keep='first')]
    
    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='H')

    solar_series = df[df['Source'] == 'Solar']['Production'].reindex(full_index, fill_value=0)
    wind_series = df[df['Source'] == 'Wind']['Production'].reindex(full_index, fill_value=0)
    combined_series = (solar_series + wind_series)
    
    datasets = {
        'Solar': solar_series.to_frame('Production'),
        'Wind': wind_series.to_frame('Production'),
        'Combined': combined_series.to_frame('Production')
    }
    
    analyzed_dfs = []
    kpi_data = {}

    for name, source_df in datasets.items():
        source_df.sort_index(inplace=True)
        source_df['Absolute Change'] = source_df['Production'].diff()
        source_df['7d Rolling Std'] = source_df['Production'].rolling(window=7*24).std()
        
        kpi_data[f'{name.lower()}_volatility'] = f"{source_df['Absolute Change'].std():.4f}"
        analyzed_dfs.append(source_df.add_prefix(f'{name}_'))

    fluctuation_df = pd.concat(analyzed_dfs, axis=1)
    fluctuation_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'fluctuation_data.csv')
    fluctuation_df.to_csv(fluctuation_filepath)
    session['fluctuation_file_path'] = fluctuation_filepath
    
    chart_data = {
        'labels': full_index.strftime('%Y-%m-%d').tolist(),
        'solar_production': datasets['Solar']['Production'].tolist(),
        'wind_production': datasets['Wind']['Production'].tolist(),
        'solar_7d_std': datasets['Solar']['7d Rolling Std'].fillna(0).tolist(),
        'wind_7d_std': datasets['Wind']['7d Rolling Std'].fillna(0).tolist(),
        'combined_7d_std': datasets['Combined']['7d Rolling Std'].fillna(0).tolist()
    }
    
    return render_template('dashboard.html', chart_data=chart_data, kpi_data=kpi_data)

@app.route('/download/fluctuation_data')
def download_fluctuation_data():
    fluctuation_file = session.get('fluctuation_file_path')
    if fluctuation_file and os.path.exists(fluctuation_file):
        return send_file(fluctuation_file, as_attachment=True, download_name='fluctuation_data.csv')
    return redirect(url_for('dashboard'))

if __name__ == '__main__':
    app.run(debug=True)