# Core IDs
ROUTE_ID = "Route ID" 
VEHICLE_ID = "Truck ID"
CUSTOMER_ID = "Customer ID"
DATE = "Date"

# Raw Numeric Features
DISTANCE_KM = "KM Traveled"
LITERS = "Liters"
FUEL = "Fuel"
MAINTENANCE = "Maintenance"
FIXED_COSTS = "Fixed Costs"
NET_REVENUE = "Net Revenue"
WEIGHT_KG = "Weight (Kg)"
WEIGHT_CUBIC = "Weight (Cubic)"
GOODS_VALUE = "Goods Value"

# Calculated Features
TOTAL_COST = "Total Costs" # Renamed from "Total Cost" to match user code "Total Costs" if strictly needed, but standardizing. User used 'Total Costs'
COST_PER_KM = "Costs per KM"
KM_PER_LITER = "KM per Liter"
MAINTENANCE_PER_KM = "Maintenance per KM"
COSTS_PER_KG = "Costs per kg"
REVENUE_PER_KM = "Revenue per KM"
REVENUE_PER_KG = "Revenue per kg"
FUEL_COSTS_PER_KM = "Fuel costs per KM"
FIXED_COSTS_PER_KM = "Fixed costs per KM"
NET_PROFIT = "Net Profit"

# Vehicle
TRUCK_TYPE = "Truck Type"

# Log features
LOG_NET_PROFIT = "Log Net Profit" # If we want to log the target

CATEGORICAL_FEATURES = [
    TRUCK_TYPE
]

# All numerical features required for the model
NUMERICAL_FEATURES = [
    DISTANCE_KM, LITERS, FUEL, MAINTENANCE, FIXED_COSTS, NET_REVENUE,
    WEIGHT_KG, WEIGHT_CUBIC, GOODS_VALUE, KM_PER_LITER, MAINTENANCE_PER_KM,
    TOTAL_COST, COST_PER_KM, COSTS_PER_KG, REVENUE_PER_KM, REVENUE_PER_KG,
    FUEL_COSTS_PER_KM, FIXED_COSTS_PER_KM
]

TARGET = NET_PROFIT
