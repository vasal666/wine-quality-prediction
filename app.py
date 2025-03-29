from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
from io import StringIO

app = Flask(__name__)

# Load model once at startup
model = joblib.load('wine_quality_model.pkl')
scaler = joblib.load('scaler.pkl')

def validate_inputs(data):
    """Validate all inputs are positive numbers"""
    for key, value in data.items():
        if float(value) < 0:
            raise ValueError(f"Negative value not allowed for {key}")
    return True

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Validate inputs
        validate_inputs(request.form)
        
        # Convert to DataFrame for easier handling
        features = pd.DataFrame([request.form])
        
        # Convert to float and scale
        X = features.astype(float)
        X_scaled = scaler.transform(X)
        
        # Predict
        prediction = model.predict(X_scaled)[0]
        result = "GOOD WINE! 🍷 (Quality ≥ 7)" if prediction == 1 else "AVERAGE WINE 😐 (Quality < 7)"
        
        return render_template('index.html', 
                            prediction_text=result,
                            original_values=request.form)
    
    except Exception as e:
        return render_template('index.html', 
                            prediction_text=f"Error: {str(e)}",
                            original_values=request.form)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)