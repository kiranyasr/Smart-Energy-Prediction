from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load saved model + scaler
model = joblib.load("energy_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # Extract features (must match training order)
    features = np.array([[
        data["Building_Floor_Area"],
        data["Building_Type"],
        data["Previous_Hour_kWh"],
        data["Occupancy_Count"],
        data["Day_of_Week"],
        data["Outdoor_Temperature"],
        data["Humidity"],
        data["Solar_Irradiance"],
        data["Hour_of_Day"],
        data["HVAC_SetPoint"],
        data["Holiday_Indicator"],
        data["Seasonal_Index"],
        data["Season_Sin"],
        data["Season_Cos"],
        data["Tomorrow_Temp_Forecast"],
        data["Tomorrow_Solar_Forecast"],
        data["Real_Time_Occupancy"]
    ]])

    # Scale
    features_scaled = scaler.transform(features)

    # Predict
    predicted = model.predict(features_scaled)[0]

    # Calculate wastage (example baseline)
    actual = data.get("Actual_Consumption", predicted)  # optional if real data available
    wastage = actual - predicted

    return jsonify({
        "Predicted_Consumption": round(float(predicted), 2),
        "Estimated_Wastage": round(float(wastage), 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
