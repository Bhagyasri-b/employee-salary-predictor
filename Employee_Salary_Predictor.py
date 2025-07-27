
# Employee Salary Predictor - Jupyter Notebook

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Sample dataset
data = {
    'Experience': [1, 3, 5, 7, 9],
    'Education': ['Bachelor', 'Master', 'PhD', 'Master', 'Bachelor'],
    'Role': ['Data Analyst', 'Data Scientist', 'ML Engineer', 'Software Dev', 'HR'],
    'Location': ['Hyderabad', 'Bangalore', 'Chennai', 'Hyderabad', 'Delhi'],
    'Salary': [300000, 700000, 1200000, 850000, 400000]
}
df = pd.DataFrame(data)

# One-hot encoding
df_encoded = pd.get_dummies(df, columns=['Education', 'Role', 'Location'])

# Split data
X = df_encoded.drop("Salary", axis=1)
y = df_encoded["Salary"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("R² Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

# Save model
import joblib
joblib.dump(model, "salary_predictor_model.pkl")
