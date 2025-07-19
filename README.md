## Loan-Approval-Predictor-Python-Project

# Loan Approval Predictor
This project aims to predict whether a loan application will be approved based on various applicant attributes using machine learning techniques. It focuses on binary classification using logistic regression and decision tree algorithms.

# Objective
The main goal of this project is to automate the decision-making process in loan approval by analyzing key features such as income, loan amount, credit score, education level, and self-employment status. This simulation reflects real-world practices in FinTech and banking systems.

# Dataset
The dataset was collected and stored in Google Sheets.
Dataset Link : https://docs.google.com/spreadsheets/d/1emgcsuxnjnpwzNzAztgSh7yloj543Mktk9Wd-m8Fujo/edit?usp=sharing
The dataset includes features like:
Applicant Income
Loan Amount
Credit Score
Education Level
Self-Employed (Yes/No)
Loan Approval Status (Target Variable)

# Models Used
Two machine learning models were trained and evaluated:
Logistic Regression
Decision Tree Classifier

These models were selected for their interpretability and suitability for binary classification problems.

# Evaluation Metrics
Both models were evaluated using the following metrics:
Accuracy
Precision
Recall
F1 Score
ROC AUC Score

At the end of the training phase, the model performances were compared, and a recommendation was made based on the overall metrics.

# Statistical Analysis
To ensure the robustness of the features used, the following statistical tests were performed:

Normality tests using Shapiro-Wilk and D’Agostino & Pearso
T-Test and Mann–Whitney U Test for mean comparison
Levene’s Test for variance equality
Kolmogorov–Smirnov Test for distribution differences
Chi-Square Test for categorical variable association
Binomial Test for approval rate proportion
Z-score-based outlier detection

# Exploratory Data Analysis
The project includes comprehensive data visualization and EDA, using:
Count plots for approval status
Scatter plots to identify trends between variables
Violin and box plots to observe distributions
Correlation heatmaps for feature relationships

# Tools & Libraries
The following technologies and libraries were used:
Python, Pandas, numpy, matplotlib, seaborn, scikit-learn, scipy, statsmodels

🧠 Real-Life Application
This project is a great example of how machine learning is used in the financial industry to improve and automate credit risk assessment. Automating loan approvals can speed up the process, reduce bias, and improve financial inclusion.
