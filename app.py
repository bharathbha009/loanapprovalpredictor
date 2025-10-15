
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model pipeline
model = pickle.load(open('loan_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # form fields
    form = request.form
    # Collect inputs with names matching the template
    input_df = pd.DataFrame([{
        'Gender': form.get('Gender'),
        'Married': form.get('Married'),
        'Dependents': form.get('Dependents'),
        'Education': form.get('Education'),
        'Self_Employed': form.get('Self_Employed'),
        'ApplicantIncome': float(form.get('ApplicantIncome') or 0),
        'CoapplicantIncome': float(form.get('CoapplicantIncome') or 0),
        'LoanAmount': float(form.get('LoanAmount') or 0),
        'Loan_Amount_Term': float(form.get('Loan_Amount_Term') or 360),
        'Credit_History': float(form.get('Credit_History') or 1.0),
        'Property_Area': form.get('Property_Area')
    }])
    pred = model.predict(input_df)[0]
    prob = float(model.predict_proba(input_df)[0][1])
    result = "Approved" if pred == 1 else "Rejected"
    return render_template('index.html', result=result, prob=f"{prob:.2f}", data=input_df.to_dict(orient='records')[0])

if __name__ == '__main__':
    app.run(debug=True)
