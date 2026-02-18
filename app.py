from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load your trained models
logistic_model = pickle.load(open('logistic_model.pkl', 'rb'))
rf_model = pickle.load(open('rf_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    features = [request.form.get('feature1'), request.form.get('feature2')]
    
    # Make predictions
    logistic_pred = logistic_model.predict([features])[0]  # 0 or 1
    rf_pred = rf_model.predict([features])[0]  # 0 or 1
    
    # Get probabilities
    logistic_proba = logistic_model.predict_proba([features])[0]
    rf_proba = rf_model.predict_proba([features])[0]
    
    return render_template('index.html', 
                         logistic_pred=logistic_pred,
                         rf_pred=rf_pred,
                         logistic_proba=logistic_proba,
                         rf_proba=rf_proba)

if __name__ == '__main__':
    app.run(debug=True)
