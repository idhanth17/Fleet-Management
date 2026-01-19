# predict.py

import joblib
import pandas as pd
from config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES

import streamlit as st
import joblib
import pandas as pd
import os
from config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES

@st.cache_resource
def get_pipeline():
    """
    Robust pipeline loader. 
    1. Tries to load from disk.
    2. If fails (AttributeError/Version mismatch), Retrains model immediately.
    """
    model_path = "data/model_pipeline.pkl"
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError("Model not found")
        return joblib.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}. Retraining...")
        from train_model import train_pipeline
        return train_pipeline()

def predict_cost(input_data):
    pipeline = get_pipeline()
    
    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])
    else:
        input_df = input_data
        
    return pipeline.predict(input_df)[0]
