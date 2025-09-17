from flask import Flask, render_template, request, session, redirect, url_for, send_file
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
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
            session['original_filepath'] = original_filepath
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
    
    datasets = { 'Solar': solar_series.to_frame('Production'), 'Wind': wind_series.to_frame('Production'), 'Combined': combined_series.to_frame('Production') }
    
    kpi_data = {}
    for name, source_df in datasets.items():
        kpi_data[f'{name.lower()}_volatility'] = f"{source_df['Production'].diff().std():.4f}"
    
    chart_data = {
        'labels': full_index.strftime('%Y-%m-%d').tolist(),
        'solar_production': datasets['Solar']['Production'].tolist(),
        'wind_production': datasets['Wind']['Production'].tolist(),
        'solar_7d_std': datasets['Solar']['Production'].rolling(window=7*24).std().fillna(0).tolist(),
        'wind_7d_std': datasets['Wind']['Production'].rolling(window=7*24).std().fillna(0).tolist(),
        'combined_7d_std': datasets['Combined']['Production'].rolling(window=7*24).std().fillna(0).tolist()
    }
    
    return render_template('dashboard.html', chart_data=chart_data, kpi_data=kpi_data)

# --- Day 3: Feature Engineering ---
def engineer_features(df_source):
    source_df = df_source.copy()
    source_df['prod_lag_24h'] = source_df['Production'].shift(24)
    source_df['rolling_mean_24h'] = source_df['Production'].rolling(window=24).mean()
    source_df['rolling_std_24h'] = source_df['Production'].rolling(window=24).std()
    
    timestamp_s = source_df.index.map(pd.Timestamp.timestamp)
    day_secs = 24 * 60 * 60
    year_secs = 365.2425 * day_secs
    source_df['sin_day'] = np.sin(2 * np.pi * timestamp_s / day_secs)
    source_df['cos_day'] = np.cos(2 * np.pi * timestamp_s / day_secs)
    source_df['sin_year'] = np.sin(2 * np.pi * timestamp_s / year_secs)
    source_df['cos_year'] = np.cos(2 * np.pi * timestamp_s / year_secs)
    
    source_df.dropna(inplace=True)
    return source_df

@app.route('/feature_engineering')
def feature_engineering():
    original_file = session.get('original_filepath')
    if not original_file or not os.path.exists(original_file):
        return redirect(url_for('upload_page'))

    df = pd.read_csv(original_file, parse_dates=['Date and Hour'], index_col='Date and Hour')
    df = df[~df.index.duplicated(keep='first')]
    
    engineered_data = {}

    for source_name in ['Solar', 'Wind']:
        source_df = engineer_features(df[df['Source'] == source_name].copy())
        
        features = ['prod_lag_24h', 'rolling_mean_24h', 'rolling_std_24h', 
                    'sin_day', 'cos_day', 'sin_year', 'cos_year']
        X = source_df[features]
        y = source_df['Production']
        
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)
        
        importance = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
        importance = importance.sort_values('importance', ascending=False)
        
        engineered_data[source_name] = { 'importance': importance.to_dict(orient='records') }

    return render_template('feature_engineering.html', engineered_data=engineered_data)

# --- Day 4: Feature Selection & Extraction ---
@app.route('/feature_selection')
def feature_selection():
    original_file = session.get('original_filepath')
    if not original_file or not os.path.exists(original_file):
        return redirect(url_for('upload_page'))

    df = pd.read_csv(original_file, parse_dates=['Date and Hour'], index_col='Date and Hour')
    df = df[~df.index.duplicated(keep='first')]
    
    solar_df = df[df['Source'] == 'Solar'].copy()
    engineered_df = engineer_features(solar_df)
    
    features = ['prod_lag_24h', 'rolling_mean_24h', 'rolling_std_24h', 
                'sin_day', 'cos_day', 'sin_year', 'cos_year']
    X = engineered_df[features]
    y = engineered_df['Production']

    corr_matrix = X.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
    X_selected = X.drop(columns=to_drop)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_selected, y)
    importance = pd.DataFrame({
        'feature': X_selected.columns, 
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_selected)
    pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
    pca_df['production'] = y.values
    
    page_data = {
        'importance': importance.to_dict(orient='records'),
        'pca_data': pca_df.to_dict(orient='records'),
        'selected_features': X_selected.columns.tolist(),
        'dropped_features': to_drop
    }
    
    return render_template('feature_selection.html', page_data=page_data)

# The download function has been removed.

if __name__ == '__main__':
    app.run(debug=True)