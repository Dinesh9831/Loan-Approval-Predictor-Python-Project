import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import joblib

print("Starting training process...")

# Load data
df = pd.read_csv('loan_approval.csv')

# Preprocess target
# Drop rows where LoanApproved is missing or NaN
df = df.dropna(subset=['LoanApproved'])
# Consider values >= 0.5 as 1 and < 0.5 as 0
df['LoanApproved'] = (df['LoanApproved'] >= 0.5).astype(int)

# Preprocess features
# Categorical features
df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
df['SelfEmployed'] = df['SelfEmployed'].map({'Yes': 1, 'No': 0})

X = df[['ApplicantIncome', 'LoanAmount', 'CreditScore', 'Education', 'SelfEmployed']]
y = df['LoanApproved']

# Handle missing values in features using median imputation
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# Train the Random Forest Model
clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
clf.fit(X_imputed, y)

# Save the trained model and imputer
joblib.dump(clf, 'model.pkl')
joblib.dump(imputer, 'imputer.pkl')

print("Model and imputer trained and saved successfully.")
