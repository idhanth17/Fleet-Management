# predict.py

import joblib
import pandas as pd
from config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES

pipeline = joblib.load("data/model_pipeline.pkl")

def predict_cost(input_data):
    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])
    else:
        input_df = input_data
        
    return pipeline.predict(input_df)[0]
