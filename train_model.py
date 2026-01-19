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

def train_pipeline():
    # Load and Preprocess
    vehicles, customers, f_cost, f_freight = load_raw_data()
    df, _ = preprocess_data(vehicles, customers, f_cost, f_freight)

    X = df[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
    y = df[TARGET]

    # Define Preprocessor
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
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    
    print(f"R2 Score: {r2:.4f}")
    
    # Save
    joblib.dump(pipeline, "data/model_pipeline.pkl")
    print("Model saved to data/model_pipeline.pkl")
    
    return pipeline

if __name__ == "__main__":
    train_pipeline()
