# data_loader.py

import pandas as pd
import streamlit as st

@st.cache_data
def load_raw_data():
    """
    Loads all raw datasets used in the Colab notebook.
    """

    # Dimension tables
    dimension_tables = pd.read_excel(
        "data/DimensionTables.xlsx"
    )

    vehicles = pd.read_excel(
        "data/DimensionTables.xlsx",
        sheet_name="Vehicles"
    )

    customers = pd.read_excel(
        "data/DimensionTables.xlsx",
        sheet_name="Customers"
    )

    # Cost & freight data
    f_cost = pd.read_excel(
        "data/fCosts.xlsx",
        header=2
    )

    f_freight = pd.read_csv(
        "data/fFreight.csv"
    )

    return vehicles, customers, f_cost, f_freight
