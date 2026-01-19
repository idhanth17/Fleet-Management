# eda.py

import matplotlib.pyplot as plt
import seaborn as sns
from config import TRUCK_TYPE, COST_PER_KM

def plot_truck_type_analysis_bar(df):
    """
    Plots a bar chart of mean costs (Cost, Maintenance, Fuel, Fixed) per KM by Truck Type.
    Corresponds to user's: truck_type_analysis.plot(kind = 'bar')
    """
    cols = ['Costs per KM', 'Maintenance per KM', 'Fuel costs per KM', 'Fixed costs per KM']
    # Group by Truck Type and mean
    truck_type_analysis = df.groupby(TRUCK_TYPE)[cols].mean()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    truck_type_analysis.plot(kind='bar', ax=ax)
    ax.set_title("Average Costs per KM by Truck Type")
    ax.set_ylabel("Cost")
    plt.xticks(rotation=45)
    return fig

def plot_costs_per_km_boxplot(df):
    """
    Plots a boxplot of Costs per KM by Truck Type.
    Corresponds to user's: sns.boxplot(..., x='Truck Type', y='Costs per KM', ...)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=df, 
        x=TRUCK_TYPE, 
        y=COST_PER_KM, 
        palette='coolwarm', 
        hue=TRUCK_TYPE, 
        ax=ax
    )
    # User's Title: 'Costs per KM by Truck Type'
    ax.set_title('Costs per KM by Truck Type', fontsize=16)
    ax.set_xlabel('Truck Type', fontsize=12)
    ax.set_ylabel('Costs per KM', fontsize=12)
    return fig

def plot_top_10_cities(city_stats):
    """
    Plots a barplot of Top 10 Cities by Goods Value.
    Corresponds to user's: sns.barplot(data=top_10_cities, x='City', y='Goods Value', ...)
    """
    # Logic to get top 10
    if city_stats.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No City Data Available")
        return fig

    # f_details_CustomerID['Goods Value'] = ...astype(int)
    # top_10_cities = ...sort_values(...).drop_duplicates().head(10)
    city_stats['Goods Value'] = city_stats['Goods Value'].astype(int)
    # The user's code re-sorts and drops duplicates. Since city_stats is already grouped by City, duplicates shouldn't exist unless name collision?
    # We follow user logic:
    top_10 = city_stats[['City', 'Goods Value']].sort_values(by='Goods Value', ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=top_10, 
        x='City', 
        y='Goods Value', 
        palette='coolwarm', 
        hue='City', 
        ax=ax
    )
    ax.set_title('Top 10 Cities by Total Goods Value', fontsize=16)
    ax.set_xlabel('City', fontsize=12)
    ax.set_ylabel('Total Goods Value', fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    return fig

def plot_geo_distribution(city_stats):
    """
    Plots a scatter plot of geographic distribution.
    Corresponds to user's: plt.scatter(..., 'Longitude', 'Latitude', ...)
    """
    if city_stats.empty or 'Latitude' not in city_stats.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No Geo Data Available")
        return fig

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        city_stats['Longitude'], 
        city_stats['Latitude'], 
        alpha=0.6, 
        color='teal', 
        edgecolor='black'
    )
    ax.set_title('Geographic Distribution of Customers', fontsize=16)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    return fig
