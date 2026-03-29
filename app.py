from flask import Flask, request, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model and imputer
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
IMPUTER_PATH = os.path.join(os.path.dirname(__file__), 'imputer.pkl')

try:
    model = joblib.load(MODEL_PATH)
    imputer = joblib.load(IMPUTER_PATH)
except FileNotFoundError:
    print("Warning: Model files not found. Please run train_model.py first.")
    model = None
    imputer = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not imputer:
        return render_template('index.html', prediction_text="Error: Model missing. Train the model first.", status_class="rejected")

    if request.method == 'POST':
        try:
            ApplicantIncome = float(request.form['ApplicantIncome'])
            LoanAmount = float(request.form['LoanAmount'])
            CreditScore = float(request.form['CreditScore'])
            Education = int(request.form['Education'])
            SelfEmployed = int(request.form['SelfEmployed'])
            
            # Feature vector
            features = np.array([[ApplicantIncome, LoanAmount, CreditScore, Education, SelfEmployed]])
            
            # Impute specific values if missing
            features_imputed = imputer.transform(features)
            
            # Prediction
            prediction = model.predict(features_imputed)[0]
            
            if prediction == 1:
                result = "Congratulations! Your Loan is Approved ✅"
                status_class = "approved"
            else:
                result = "Sorry, Your Loan is Rejected ❌"
                status_class = "rejected"
            
            return render_template('index.html', prediction_text=result, status_class=status_class)
            
        except Exception as e:
            return render_template('index.html', prediction_text=f"Error: {str(e)}", status_class="rejected")

if __name__ == "__main__":
    app.run(debug=True, port=5000)
