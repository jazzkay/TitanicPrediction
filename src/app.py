# src/app.py
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'rf_model.joblib')

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'), static_folder=os.path.join(BASE_DIR, 'static'))

if os.path.exists(MODEL_PATH):
    pipeline = joblib.load(MODEL_PATH)
else:
    pipeline = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if pipeline is None:
        return jsonify({'error': 'Model not found. Train model first.'}), 400
    data = request.json
    df = pd.DataFrame([data])
    pred = pipeline.predict(df)[0]
    proba = pipeline.predict_proba(df).max() if hasattr(pipeline, 'predict_proba') else None
    return jsonify({'survived': int(pred), 'confidence': float(proba) if proba is not None else None})

if __name__ == '__main__':
    app.run(debug=True)
