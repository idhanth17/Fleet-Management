import numpy as np
import pandas as pd
from config import (
    ROUTE_ID, VEHICLE_ID, CUSTOMER_ID, TOTAL_COST, DISTANCE_KM, 
    COST_PER_KM, DATE, FUEL, MAINTENANCE, FIXED_COSTS,
    LITERS, NET_REVENUE, WEIGHT_KG, WEIGHT_CUBIC, GOODS_VALUE,
    KM_PER_LITER, MAINTENANCE_PER_KM, COSTS_PER_KG, REVENUE_PER_KM,
    REVENUE_PER_KG, FUEL_COSTS_PER_KM, FIXED_COSTS_PER_KM, NET_PROFIT
)
def preprocess_data(vehicles, customers, f_cost, f_freight):
    # Prevent SettingWithCopyWarning by working on explicit copies
    f_cost = f_cost.copy()
    f_freight = f_freight.copy()
    vehicles = vehicles.copy()
    customers = customers.copy()

    # --- User provided cleaning logic ---
    
    # --- User provided cleaning logic ---
    
    # Remove repeated headers in data if any
    if 'KM Traveled' in f_cost.columns:
         f_cost = f_cost[f_cost['KM Traveled'].astype(str) != 'KM Traveled'].copy()

    # Safe cleaning helper
    def clean_num(df, col, to_type=float, replace_comma_dot=False, replace_comma_empty=False):
        if col not in df.columns: return pd.Series(0, index=df.index)
        s = df[col].astype(str)
        if replace_comma_dot:
            s = s.str.replace(',', '.')
        if replace_comma_empty:
            s = s.str.replace(',', '')
        # Handle 'nan' string
        s = s.replace('nan', '0')
        return pd.to_numeric(s, errors='coerce').fillna(0).astype(to_type)

    # f_cost Cleaning
    f_cost['KM Traveled'] = clean_num(f_cost, 'KM Traveled', int)
    f_cost['Liters'] = clean_num(f_cost, 'Liters', float)
    f_cost['Fuel'] = clean_num(f_cost, 'Fuel', float)
    f_cost['Maintenance'] = clean_num(f_cost, 'Maintenance', float)
    f_cost['Fixed Costs'] = clean_num(f_cost, 'Fixed Costs', float)
    
    # Ensure Date
    f_cost['Date'] = pd.to_datetime(f_cost['Date'], errors='coerce')


    # f_freight (f_details_1) Cleaning
    # User: replace(',', '.') for Net Revenue, Weight. replace(',', '') for Goods Value.
    f_freight['Net Revenue'] = clean_num(f_freight, 'Net Revenue', float, replace_comma_dot=True)
    f_freight['Weight (Kg)'] = clean_num(f_freight, 'Weight (Kg)', float, replace_comma_dot=True)
    f_freight['Weight (Cubic)'] = clean_num(f_freight, 'Weight (Cubic)', float, replace_comma_dot=True)
    f_freight['Goods Value'] = clean_num(f_freight, 'Goods Value', float, replace_comma_empty=True)
    
    f_freight['Date'] = pd.to_datetime(f_freight['Date'], errors='coerce')
    f_freight['Net Revenue'] = f_freight['Net Revenue'] * 1000 # User logic
    
    # Dimensional Joins
    # Vehicles (dimension_tab2) on Truck ID
    # Note: Column names in Vehicles might have whitespace, handled below
    
    # Freight Flow
    # f_details_2 = f_details_1.merge(dimension_tab2,how = 'left', on = 'Truck ID' )
    f_details_2 = f_freight.merge(vehicles, how='left', on='Truck ID')
    
    # f_details_3 = f_details_2.merge(dimension_tab3,how = 'left', on = 'Customer ID' )
    f_details_3 = f_details_2.merge(customers, how='left', on='Customer ID')
    
    # Rename and Drop
    # f_details_4 = f_details_3.rename(columns={"Year_x": "Year", "Year_y": "Truck Age" , 'City_x' : 'City'})
    # Need to check if these columns exist. Assuming standard schema.
    renames = {"Year_x": "Year", "Year_y": "Truck Age", 'City_x': 'City'}
    f_details_4 = f_details_3.rename(columns=renames)
    if 'City_y' in f_details_4.columns:
        f_details_4.drop('City_y', axis=1, inplace=True)
        
    # --- Aggregation for Model (Truck Level) ---
    
    # f_cost_3 = f_cost_2.merge(dimension_tab2, how = 'left', on = 'Truck ID')
    f_cost_3 = f_cost.merge(vehicles, how='left', on='Truck ID')
    
    # f_cost_4 rename Drive ID -> Driver ID 
    # (Skip Driver join as it's not used for main model dataframe, only for a specific driver analysis if needed)
    f_cost_4 = f_cost_3 # simplify
    
    # Group By
    # f_cost_TruckID = ... sum()
    cost_cols = ['KM Traveled', 'Liters', 'Fuel', 'Maintenance', 'Fixed Costs']
    f_cost_TruckID = f_cost_4.groupby(['Truck ID', 'Truck Type', 'Plate'])[cost_cols].sum()
    
    # f_details_TruckID = ... sum()
    rev_cols = ['Net Revenue', 'Weight (Kg)', 'Weight (Cubic)', 'Goods Value']
    f_details_TruckID = f_details_4.groupby(['Truck ID', 'Truck Type', 'Plate'])[rev_cols].sum()
    
    # merged_log = inner join
    merged_log = f_cost_TruckID.merge(f_details_TruckID, how='inner', on=['Truck ID', 'Truck Type', 'Plate'])
    merged_log = merged_log.reset_index() # make them columns
    
    # --- Normalize for Single Delivery (Per Trip) ---
    # Calculate Number of Trips per Truck from Freight Data (f_details_4)
    # Each row in f_details_4 (freight+dim) is a trip/order
    trip_counts = f_details_4.groupby(['Truck ID', 'Truck Type', 'Plate']).size().reset_index(name='Num Trips')
    
    # Merge trip counts
    merged_log = merged_log.merge(trip_counts, on=['Truck ID', 'Truck Type', 'Plate'], how='left')
    merged_log['Num Trips'] = merged_log['Num Trips'].fillna(1) # Safety
    
    # Columns to normalize (Absolute Sums -> Average Per Trip)
    absolute_cols = [
        'KM Traveled', 'Liters', 'Fuel', 'Maintenance', 'Fixed Costs', 
        'Net Revenue', 'Weight (Kg)', 'Weight (Cubic)', 'Goods Value'
    ]
    
    for col in absolute_cols:
         if col in merged_log.columns:
             merged_log[col] = merged_log[col] / merged_log['Num Trips']
             
    # --- Feature Calculations (User Logic) ---
    # Derived features (Ratios) remain mathematically similar but calculated on per-trip averages
    # e.g., (AvgCost / AvgKM) is approx (TotalCost / TotalKM) 
    
    merged_log['KM per Liter'] = merged_log['KM Traveled'] / merged_log['Liters'].replace(0, np.nan)
    merged_log['Maintenance per KM'] = merged_log['Maintenance'] / merged_log['KM Traveled'].replace(0, np.nan)
    
    # Recalculate Total Costs (Per Trip)
    merged_log['Total Costs'] = merged_log['Fuel'] + merged_log['Maintenance'] + merged_log['Fixed Costs']
    
    merged_log['Costs per KM'] = merged_log['Total Costs'] / merged_log['KM Traveled'].replace(0, np.nan)
    merged_log['Costs per kg'] = merged_log['Total Costs'] / merged_log['Weight (Kg)'].replace(0, np.nan)
    merged_log['Revenue per KM'] = merged_log['Net Revenue'] / merged_log['KM Traveled'].replace(0, np.nan)
    merged_log['Revenue per kg'] = merged_log['Maintenance'] / merged_log['Weight (Kg)'].replace(0, np.nan) 
    
    # Net Profit (Per Trip)
    merged_log['Net Profit'] = merged_log['Net Revenue'] - merged_log['Total Costs']
    
    merged_log['Fuel costs per KM'] = merged_log['Fuel'] / merged_log['KM Traveled'].replace(0, np.nan)
    merged_log['Fixed costs per KM'] = merged_log['Fixed Costs'] / merged_log['KM Traveled'].replace(0, np.nan)

    merged_log = merged_log.fillna(0)
    merged_log = merged_log.replace([np.inf, -np.inf], 0)
    
    # --- City Stats for EDA ---
    # f_details_CustomerID = f_details_4.groupby(['City'])[['Weight (Kg)', 'Weight (Cubic)', 'Goods Value']].sum().reset_index()
    if 'City' in f_details_4.columns:
        city_stats = f_details_4.groupby(['City'])[['Weight (Kg)', 'Weight (Cubic)', 'Goods Value']].sum().reset_index()
        # Merge with city geo (dimension_tab3_city_group)
        # dimension_tab3 (customers)
        # dimension_tab3_city_group = dimension_tab3.groupby(['City','Latitude', 'Longitude'])['Customer ID'].nunique().reset_index()
        if 'Latitude' in customers.columns and 'Longitude' in customers.columns:
             geo_stats = customers.groupby(['City', 'Latitude', 'Longitude'])['Customer ID'].nunique().reset_index()
             city_stats = city_stats.merge(geo_stats, on='City', how='left')
    else:
        city_stats = pd.DataFrame()

    return merged_log, city_stats
