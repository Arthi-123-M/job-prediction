from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load models at startup
print("Loading models...")
logistic_model = joblib.load('logistic_model.pkl')
rf_model = joblib.load('rf_model.pkl')
feature_names = joblib.load('feature_names.pkl')
print("Models loaded successfully!")

@app.route('/')
def home():
    return render_template('index.html', 
                         logistic_pred=None, 
                         rf_pred=None,
                         logistic_fake=None,
                         logistic_real=None,
                         rf_fake=None,
                         rf_real=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get features from form
        features = []
        
        # Based on your training data features
        features.append(float(request.form.get('has_salary', 0)))
        features.append(float(request.form.get('has_company_logo', 0)))
        features.append(float(request.form.get('has_questions', 0)))
        features.append(float(request.form.get('title_length', 0)))
        features.append(float(request.form.get('description_length', 0)))
        features.append(float(request.form.get('requirements_length', 0)))
        features.append(float(request.form.get('has_telecommuting', 0)))
        features.append(float(request.form.get('has_healthcare', 0)))
        features.append(float(request.form.get('has_401k', 0)))
        features.append(float(request.form.get('has_bonus', 0)))
        
        # Convert to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)
        
        # Make predictions
        logistic_pred = logistic_model.predict(features_array)[0]
        rf_pred = rf_model.predict(features_array)[0]
        
        # Get probabilities
        logistic_proba = logistic_model.predict_proba(features_array)[0]
        rf_proba = rf_model.predict_proba(features_array)[0]
        
        # Calculate confidence percentages
        logistic_fake = round(logistic_proba[0] * 100, 1)  # 0 = Fake
        logistic_real = round(logistic_proba[1] * 100, 1)  # 1 = Real
        
        rf_fake = round(rf_proba[0] * 100, 1)
        rf_real = round(rf_proba[1] * 100, 1)
        
        return render_template('index.html',
                             logistic_pred=logistic_pred,
                             rf_pred=rf_pred,
                             logistic_fake=logistic_fake,
                             logistic_real=logistic_real,
                             rf_fake=rf_fake,
                             rf_real=rf_real,
                             features=features)
    
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
