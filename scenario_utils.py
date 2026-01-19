import pandas as pd
import numpy as np
from config import (
    NUMERICAL_FEATURES, DISTANCE_KM, LITERS, FUEL, MAINTENANCE, FIXED_COSTS,
    NET_REVENUE, WEIGHT_KG, WEIGHT_CUBIC, GOODS_VALUE, KM_PER_LITER,
    MAINTENANCE_PER_KM, TOTAL_COST, COST_PER_KM, COSTS_PER_KG, REVENUE_PER_KM,
    REVENUE_PER_KG, FUEL_COSTS_PER_KM, FIXED_COSTS_PER_KM, TRUCK_TYPE
)

def create_scenario_input(scenario_data, truck_type_means):
    """
    Creates a DataFrame for a new scenario, filling missing numerical features
    based on Truck Type averages, and recalculating derived features.
    
    Adapted from user provided logic.
    """
    input_df_row = pd.DataFrame([scenario_data])

    # Determine the base for filling in data
    # We assume scenario_data has 'Truck Type'
    if TRUCK_TYPE in scenario_data and scenario_data[TRUCK_TYPE] in truck_type_means.index:
        # Use truck type's average data as a base
        base_data = truck_type_means.loc[scenario_data[TRUCK_TYPE]].copy()
        input_df_row[TRUCK_TYPE] = scenario_data[TRUCK_TYPE]
    else:
        # Fallback if unknown truck type or not provided (should handle in UI)
        # For now, if missing, we can't easily fill defaults without a type. 
        # But let's assume UI enforces it.
        pass

    # Fill in numerical features from base_data if not explicitly provided
    for col in NUMERICAL_FEATURES:
        if col not in input_df_row.columns or pd.isna(input_df_row.loc[0, col]):
            if col in base_data:
                 input_df_row[col] = base_data[col]
            else:
                 input_df_row[col] = 0 # Safety fallback

    # Recalculate derived features based on potentially new base values
    # Ensure required columns are numeric and handle potential division by zero
    
    # helper to safe get and fill
    def get_val(col, default=0):
        return pd.to_numeric(input_df_row[col], errors='coerce').fillna(default)

    km = get_val(DISTANCE_KM, 0)
    liters = get_val(LITERS, 1) # avoid div by zero default
    fuel = get_val(FUEL, 0)
    maint = get_val(MAINTENANCE, 0)
    fixed = get_val(FIXED_COSTS, 0)
    net_rev = get_val(NET_REVENUE, 0)
    weight = get_val(WEIGHT_KG, 1) # avoid div by zero default

    # Recalculate derived features
    input_df_row[KM_PER_LITER] = km / liters
    input_df_row[MAINTENANCE_PER_KM] = maint / km.replace(0, np.nan)
    
    # Recalculate Total Costs
    input_df_row[TOTAL_COST] = fuel + maint + fixed
    
    total_cost = input_df_row[TOTAL_COST]
    
    input_df_row[COST_PER_KM] = total_cost / km.replace(0, np.nan)
    input_df_row[COSTS_PER_KG] = total_cost / weight.replace(0, np.nan)
    input_df_row[REVENUE_PER_KM] = net_rev / km.replace(0, np.nan)
    # Using Net Revenue / Weight for Revenue per kg
    input_df_row[REVENUE_PER_KG] = net_rev / weight.replace(0, np.nan)
    
    input_df_row[FUEL_COSTS_PER_KM] = fuel / km.replace(0, np.nan)
    input_df_row[FIXED_COSTS_PER_KM] = fixed / km.replace(0, np.nan)

    # Fill any NaNs created by division by zero with 0
    input_df_row = input_df_row.fillna(0)

    # Ensure only necessary columns for model are returned (plus identifiers if needed)
    # The model expects CATEGORICAL + NUMERICAL (preprocessor handles selection but good to be clean)
    return input_df_row
