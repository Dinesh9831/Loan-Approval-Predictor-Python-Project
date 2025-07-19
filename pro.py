# ----------------------------- 0. Imports -----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import (
    ttest_ind, mannwhitneyu, chi2_contingency, binomtest,
    ks_2samp, shapiro, normaltest, levene, zscore
)
from statsmodels.stats.weightstats import ztest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import warnings
warnings.filterwarnings("ignore")





# ----------------------------- 1. Load and Clean Data -----------------------------
df = pd.read_csv(r"C:\Users\LENOVO\Documents\문서\Cipher Python Project\loan_approval.csv",encoding='ISO-8859-1')



print("\n=== Basic Info ===")
print("\nHead:\n", df.head())
print("\nTail:\n", df.tail())
print("\nDescribe:\n", df.describe())
print("\nInfo:")
df.info()
print("\nMissing Values:\n", df.isnull().sum())


df = df.dropna(subset=['LoanApproved'])
df['ApplicantIncome'].fillna(df['ApplicantIncome'].median(), inplace=True)
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
df['CreditScore'].fillna(df['CreditScore'].median(), inplace=True)

df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
df['SelfEmployed'] = df['SelfEmployed'].map({'Yes': 1, 'No': 0})
df['LoanApproved'] = df['LoanApproved'].astype(int)

df.to_csv("loan_approval_cleaned.csv", index=False)


# ----------------------------- 2. Exploratory Data Analysis -----------------------------
print("\nData Info:")
print(df.info())
print("\nStatistical Summary:\n", df.describe())
print("\nLoan Approval Percentages:\n", df['LoanApproved'].value_counts(normalize=True) * 100)

# Count plots
sns.countplot(x='LoanApproved', data=df)
plt.title("Loan Approval Count")
plt.xticks([0, 1], ['Not Approved', 'Approved'])
plt.show()

sns.countplot(x='Education', hue='LoanApproved', data=df)
plt.title("Loan Approval by Education")
plt.xticks([0, 1], ['Not Graduate', 'Graduate'])
plt.show()



# Scatter plots
sns.scatterplot(data=df, x='ApplicantIncome', y='LoanAmount', hue='LoanApproved')
plt.title("Loan Amount vs Applicant Income")
plt.show()

sns.scatterplot(data=df, x='CreditScore', y='LoanAmount', hue='LoanApproved', palette='cool')
plt.title("Loan Amount vs Credit Score")
plt.show()

# Violin plots
sns.violinplot(data=df, x='LoanApproved', y='ApplicantIncome')
plt.title("Income Distribution by Loan Approval")
plt.xticks([0, 1], ['Not Approved', 'Approved'])
plt.show()

sns.violinplot(data=df, x='LoanApproved', y='CreditScore', palette="muted")
plt.title("Credit Score Distribution by Loan Approval")
plt.xticks([0, 1], ['Not Approved', 'Approved'])
plt.show()

# Box plot
sns.boxplot(data=df, x='LoanApproved', y='LoanAmount')
plt.title("Loan Amount by Loan Approval")
plt.xticks([0, 1], ['Not Approved', 'Approved'])
plt.show()

# Column (bar) plot of means
sns.barplot(data=df, x='LoanApproved', y='ApplicantIncome', estimator='mean')
plt.title("Mean Applicant Income by Loan Approval")
plt.xticks([0, 1], ['Not Approved', 'Approved'])
plt.show()

# Correlation heatmap
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()


# ----------------------------- 3. Statistical Analysis -----------------------------
approved = df[df['LoanApproved'] == 1]
not_approved = df[df['LoanApproved'] == 0]
numerical_cols = ['ApplicantIncome', 'LoanAmount', 'CreditScore']
categorical_cols = ['Education', 'SelfEmployed']

# Normality tests
print("Normality Tests")
for col in numerical_cols:
    stat_sw, p_sw = shapiro(df[col])
    stat_dp, p_dp = normaltest(df[col])
    print(f"\nFeature: {col}")
    print(f"  Shapiro-Wilk:       p = {p_sw:.4f}")
    print(f"  D’Agostino-Pearson: p = {p_dp:.4f}")

# Levene’s test for equal variances
print("\nLevene’s Variance Test")
for col in numerical_cols:
    stat, p = levene(approved[col], not_approved[col])
    print(f"{col}: p = {p:.4f}")

# T-test and Mann-Whitney U test
print("\nT-Test and Mann-Whitney U Test")
for col in numerical_cols:
    t_stat, t_p = ttest_ind(approved[col], not_approved[col], equal_var=False)
    u_stat, u_p = mannwhitneyu(approved[col], not_approved[col])
    print(f"\nFeature: {col}")
    print(f"  T-Test:         p = {t_p:.4f}")
    print(f"  Mann-Whitney U: p = {u_p:.4f}")

# Z-Test
print("\nZ-Test")
for col in numerical_cols:
    z_stat, p_val = ztest(approved[col], not_approved[col])
    print(f"{col}: z = {z_stat:.4f}, p = {p_val:.4f}")

# Kolmogorov–Smirnov test
print("\nKolmogorov–Smirnov Test")
for col in numerical_cols:
    stat, p = ks_2samp(approved[col], not_approved[col])
    print(f"{col}: p = {p:.4f}")

# Chi-Square test for categorical variables
print("\nChi-Square Test")
for col in categorical_cols:
    table = pd.crosstab(df[col], df['LoanApproved'])
    chi2, p, _, _ = chi2_contingency(table)
    print(f"{col} vs LoanApproved: p = {p:.4f}")

# Binomial test
print("\nBinomial Test")
total = len(df)
approved_count = df['LoanApproved'].sum()
binom_res = binomtest(approved_count, total, p=0.5)
print(f"Approved: {approved_count}/{total} => p = {binom_res.pvalue:.4f}")

# Z-Score for outlier detection
print("\nZ-Score Based Outlier Detection")
for col in numerical_cols:
    df[f'{col}_zscore'] = zscore(df[col])
    outliers = df[np.abs(df[f'{col}_zscore']) > 3]
    print(f"{col}: {len(outliers)} outliers (|z| > 3)")

df.drop(columns=[col for col in df.columns if '_zscore' in col], inplace=True)





# Encode categorical
df['Education'] = df['Education'].astype('category').cat.codes
df['SelfEmployed'] = df['SelfEmployed'].astype('category').cat.codes
df['LoanApproved'] = df['LoanApproved'].astype(int)

# ------------------- Z-Score Based Outlier Detection -------------------
z_scores = np.abs(zscore(df[['ApplicantIncome', 'LoanAmount', 'CreditScore']]))
outliers_count = (z_scores > 3).sum(axis=0)
print("Z-Score Based Outlier Detection")
print(outliers_count)

# ------------------- Statistical Tests -------------------
approved = df[df['LoanApproved'] == 1]
not_approved = df[df['LoanApproved'] == 0]

# Normality
print("\nShapiro-Wilk Test (ApplicantIncome):", shapiro(df['ApplicantIncome']))
print("D’Agostino & Pearson Test (ApplicantIncome):", normaltest(df['ApplicantIncome']))

# Variance Test
print("\nLevene’s Test (ApplicantIncome):", levene(approved['ApplicantIncome'], not_approved['ApplicantIncome']))

# Mean Comparison
print("\nT-test (ApplicantIncome):", ttest_ind(approved['ApplicantIncome'], not_approved['ApplicantIncome'], equal_var=False))
print("Mann-Whitney U (ApplicantIncome):", mannwhitneyu(approved['ApplicantIncome'], not_approved['ApplicantIncome']))

# Distribution Comparison
print("\nKolmogorov–Smirnov Test (ApplicantIncome):", ks_2samp(approved['ApplicantIncome'], not_approved['ApplicantIncome']))

# Categorical Association
chi2_edu, p_edu, _, _ = chi2_contingency(pd.crosstab(df['Education'], df['LoanApproved']))
chi2_se, p_se, _, _ = chi2_contingency(pd.crosstab(df['SelfEmployed'], df['LoanApproved']))
print("\nChi-Square Test - Education p-value:", p_edu)
print("Chi-Square Test - SelfEmployed p-value:", p_se)

# Binomial Test
print("\nBinomial Test for Approval Proportion (H0: 0.5):", binomtest(df['LoanApproved'].sum(), len(df), p=0.5))

# ------------------- Model Training -------------------
X = df[['ApplicantIncome', 'LoanAmount', 'CreditScore', 'Education', 'SelfEmployed']]
y = df['LoanApproved']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

# Decision Tree
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# Evaluation
def get_metrics(y_true, y_pred, y_proba):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average='weighted'),
        "Recall": recall_score(y_true, y_pred, average='weighted'),
        "F1 Score": f1_score(y_true, y_pred, average='weighted'),
        "ROC AUC": roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) == 2 else np.nan
    }

lr_metrics = get_metrics(y_test, y_pred_lr, lr.predict_proba(X_test_scaled)[:, 1])
dt_metrics = get_metrics(y_test, y_pred_dt, dt.predict_proba(X_test)[:, 1])

results_df = pd.DataFrame({
    "Metric": list(lr_metrics.keys()),
    "Logistic Regression": list(lr_metrics.values()),
    "Decision Tree": list(dt_metrics.values())
})

print("\nModel Evaluation:")
print(results_df.round(4))


# ----------------------------- 6. Final Recommendation -----------------------------
print("\nFinal Recommendation")
if lr_metrics['Accuracy'] > dt_metrics['Accuracy']:
    print("Logistic Regression performed better in terms of accuracy.")
else:
    print("Decision Tree performed better in terms of accuracy.")

if lr_metrics['ROC AUC'] > dt_metrics['ROC AUC']:
    print("Logistic Regression shows better separation based on ROC AUC.")
else:
    print("Decision Tree captures better nonlinear patterns.")

if lr_metrics['F1 Score'] > dt_metrics['F1 Score']:
    print("Use Logistic Regression for balanced performance and interpretability.")
else:
    print("Use Decision Tree if you prefer rule-based, interpretable outputs.")
