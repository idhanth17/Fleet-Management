import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from preprocessing import preprocess_data
from predict import predict_cost
from scenario_utils import create_scenario_input
from config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TRUCK_TYPE
import eda

@st.cache_data
def load_raw_data():
    """
    Loads all raw datasets.
    """
    # Dimension tables
    dimension_tables = pd.read_excel("data/DimensionTables.xlsx")
    vehicles = pd.read_excel("data/DimensionTables.xlsx", sheet_name="Vehicles")
    customers = pd.read_excel("data/DimensionTables.xlsx", sheet_name="Customers")

    # Cost & freight data
    f_cost = pd.read_excel("data/fCosts.xlsx", header=2)
    f_freight = pd.read_csv("data/fFreight.csv")

    return vehicles, customers, f_cost, f_freight

st.set_page_config(page_title="Fleet Analytics", layout="wide")
st.title("🚚 Fleet Cost Prediction System")

# Load and Preprocess Data
with st.spinner("Loading data..."):
    vehicles, customers, f_cost, f_freight = load_raw_data()
    df, city_stats = preprocess_data(vehicles, customers, f_cost, f_freight)

# Navigation
page = st.sidebar.radio("Navigate", ["Analysis", "Prediction"])

# --- TAB 1: Analysis ---
if page == "Analysis":
    st.header("Fleet Data Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Truck Type Cost Analysis")
        # Bar chart
        fig1 = eda.plot_truck_type_analysis_bar(df)
        st.pyplot(fig1)

        st.subheader("Top 10 Cities by Goods Value")
        # Top Cities
        fig3 = eda.plot_top_10_cities(city_stats)
        st.pyplot(fig3)
        
    with col2:
        st.subheader("Costs per KM Distribution")
        # Boxplot
        fig2 = eda.plot_costs_per_km_boxplot(df)
        st.pyplot(fig2)

        st.subheader("Customer Geography")
        # Geo scatter
        fig4 = eda.plot_geo_distribution(city_stats)
        st.pyplot(fig4)

# --- TAB 2: Prediction ---
if page == "Prediction":
    st.header("Profit Prediction Setup")
    
    st.markdown("Use this tool to predict **Net Profit** for a **Single Delivery**. Select a truck type to load average trip stats, then override specific values.")
    
    # 1. Inputs
    
    # Base Configuration (Outside Form to trigger update)
    c_base, _ = st.columns(2)
    with c_base:
        st.subheader("Base Configuration")
        truck_types = sorted(df[TRUCK_TYPE].dropna().unique())
        selected_truck = st.selectbox("Select Truck Type (Base)", truck_types)
    
    # Calculate means
    from config import NUMERICAL_FEATURES, DISTANCE_KM, WEIGHT_KG, WEIGHT_CUBIC, GOODS_VALUE
    truck_means = df.groupby(TRUCK_TYPE)[NUMERICAL_FEATURES].mean()

    # --- Session State Logic to Update Defaults ---
    # Initialize keys if not present
    if "trip_km" not in st.session_state:
        st.session_state.trip_km = float(truck_means.loc[selected_truck, DISTANCE_KM])
        st.session_state.weight_kg = float(truck_means.loc[selected_truck, WEIGHT_KG])
        st.session_state.weight_cubic = float(truck_means.loc[selected_truck, WEIGHT_CUBIC])
        st.session_state.goods_value = float(truck_means.loc[selected_truck, GOODS_VALUE])
        st.session_state.last_truck = selected_truck

    # Check for change
    if st.session_state.last_truck != selected_truck:
        st.session_state.trip_km = float(truck_means.loc[selected_truck, DISTANCE_KM])
        st.session_state.weight_kg = float(truck_means.loc[selected_truck, WEIGHT_KG])
        st.session_state.weight_cubic = float(truck_means.loc[selected_truck, WEIGHT_CUBIC])
        st.session_state.goods_value = float(truck_means.loc[selected_truck, GOODS_VALUE])
        st.session_state.last_truck = selected_truck
        # We don't need rerun because widgets are drawn below

    # Form for Scenario Inputs
    with st.form("scenario_form"):
        c1, c2 = st.columns(2)
        with c1:
             st.subheader("Scenario Variables")
             # Use keys to bind to session state
             km_traveled = st.number_input("Trip KM Traveled", key="trip_km", min_value=1.0)
             weight_kg = st.number_input("Weight (Kg)", key="weight_kg", min_value=1.0)
             weight_cubic = st.number_input("Weight (Cubic)", key="weight_cubic", min_value=1.0)
             goods_value = st.number_input("Goods Value", key="goods_value", min_value=0.0)

        with c2:
             st.subheader("Run Prediction")
             st.write("Adjust values on the left and click Calculate.")
             submitted = st.form_submit_button("Calculate Net Profit", type="primary")

    if submitted:
             # Logic ...
            scenario_data = {
                TRUCK_TYPE: selected_truck,
                DISTANCE_KM: km_traveled,
                WEIGHT_KG: weight_kg,
                WEIGHT_CUBIC: weight_cubic,
                GOODS_VALUE: goods_value
            }
            
            try:
                # Prepare input DF using utility
                scenario_df = create_scenario_input(scenario_data, truck_means)
                
                # Predict
                # Use predict module wrapper if updated, or load pipeline here?
                # Best to use predict module for consistency
                from predict import predict_cost 
                prediction = predict_cost(scenario_df)
                
                st.metric("Predicted Net Profit", f"₹ {prediction:,.2f}")
                
                if prediction > 0:
                    st.success("This scenario is profitable!")
                else:
                    st.error("This scenario results in a loss.")
                    
            except Exception as e:
                st.error(f"Prediction Error: {e}")

