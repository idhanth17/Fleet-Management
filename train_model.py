# train_model.py

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

from data_loader import load_raw_data
from preprocessing import preprocess_data
from config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET

# Load and Preprocess
vehicles, customers, f_cost, f_freight = load_raw_data()
df, _ = preprocess_data(vehicles, customers, f_cost, f_freight)

X = df[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
y = df[TARGET]

# Define Preprocessor
# Using 'passthrough' for numerical features as they are already scaled/computed 
# or if RF doesn't strictly need scaling (unlike Linear Regression where it helps convergence/interpretation).
# User code used 'passthrough' for numericals, so we follow that.
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ("num", StandardScaler(), NUMERICAL_FEATURES),
    ]
)

# Define Pipeline with Random Forest
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(random_state=42))
    ]
)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
print("Training Random Forest model...")
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("--- Random Forest Performance ---")
print(f"R2 Score: {r2:.4f}")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")

# Save
joblib.dump(pipeline, "data/model_pipeline.pkl")
print("Model saved to data/model_pipeline.pkl")
