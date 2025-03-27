#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np
import seaborn as sns
import gradio as gr
import joblib
import xgboost as xgb
import pickle  # To load the saved model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import plotly.graph_objects as go
import time

import warnings
warnings.filterwarnings('ignore')  # Suppress warnings




# Load transportation cost matrix
cost_df = pd.read_csv("data/transportation_cost_matrix.csv", index_col=0)
cost_df.replace(-1, float("inf"), inplace=True)

# Create a weighted graph
G = nx.Graph()
for city in cost_df.index:
    G.add_node(city)
for i in cost_df.index:
    for j in cost_df.columns:
        if i != j and cost_df.loc[i, j] != float("inf"):
            G.add_edge(i, j, weight=cost_df.loc[i, j])


# In[14]:


# Load pre-trained XGBoost regression model
with open("models/xgb_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load the feature scaler
scaler_X = joblib.load("models/scaler_X.pkl")


# In[15]:


# city mapping with city code
city_mapping = {
    0: "Delhi",
    1: "Mumbai",
    2: "Kolkata",
    3: "Jaipur",
    4: "Chandigarh",
    5: "Pune",
    6: "Bangluru",
    7: "Lucknow",
    8: "Agra",
    9: "Patna"
}


# In[16]:


# Function to animate the shortest path
def animate_shortest_path(city):
    source = "Food Hub"
    target_city = city_mapping.get(city)
    
    try:
        path = nx.dijkstra_path(G, source, target_city, weight="weight")
        cost = nx.dijkstra_path_length(G, source, target_city, weight="weight")
        
        pos = nx.spring_layout(G, seed=34)
        
        node_colors = ["red" if node == source else "pink" if node == target_city else "#00FFFF" for node in G.nodes()]
        edge_colors = ["black" for u, v in G.edges()]
        edge_widths = [5 if (u, v) in zip(path, path[1:]) or (v, u) in zip(path, path[1:]) else 1 for u, v in G.edges()]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        def update(num):
            ax.clear()
            plt.title(f"Shortest Path: {source} to {target_city} (Cost: {cost})", fontsize=30)
            nx.draw(G, pos, with_labels=False, node_color=node_colors, edge_color=edge_colors, width=edge_widths, ax=ax, node_size=3000)
            
            # Draw edge labels (costs) near the nodes
            edge_labels = nx.get_edge_attributes(G, 'weight')
            for edge, label in edge_labels.items():
                u, v = edge
                x = (pos[u][0] + pos[v][0]) / 2
                y = (pos[u][1] + pos[v][1]) / 2
                # Calculate offset to place label near node, not in the middle.
                offset_x = (pos[v][0] - pos[u][0]) * 0.15
                offset_y = (pos[v][1] - pos[u][1]) * 0.15
                ax.text(x + offset_x, y + offset_y, str(label), fontsize=10, color="black", ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            nx.draw_networkx_labels(G, pos, labels={node: node for node in G.nodes()},font_size=12, font_color="black", ax=ax)
            nx.draw_networkx_nodes(G, pos, nodelist=path[:num+1], node_color="red", ax=ax)
            nx.draw_networkx_edges(G, pos, edgelist=[(path[i], path[i+1]) for i in range(num)], width=3, edge_color="red", ax=ax)

        ani = animation.FuncAnimation(fig, update, frames=len(path), interval=500, repeat=False)

        ani.save("animated_shortest_path.gif", writer="pillow")
        return f"Optimal Path:  {' -> '.join(path)}\nTotal Cost:  {cost} units","animated_shortest_path.gif"
    except nx.NetworkXNoPath:
        return None


# In[17]:


# Define prediction function
def predict_food_demand(city_name, temp, rain, DOW, humidity, air_qua_ind,holiday,avail):
    if city_name is None:
        city_name = "Delhi"  # Default city

    if temp is None:
        temp = 18.4  # Default temperature

    if rain is None:
        rain = 4.0  # Default rainfall

    if DOW is None:
        DOW = 0  # Default to Monday

    if humidity is None:
        humidity = 54.9  # Default humidity

    if air_qua_ind is None:
        air_qua_ind = 124.9  # Default AQI

    if holiday is None:
        holiday = "No"  # Default value for holiday
        
    city = [key for key, value in city_mapping.items() if value == city_name][0]
    
    data_dict = {
        'City': city,
        'Latitude': 0,
        'Longitude': 0,
        'Temperature': temp,
        'Rainfall': rain,
        'Population Growth Factor': 0, 
        'Holiday': 0,
        'Day of Week': DOW,
        'Unemployment Rate (%)': 6.471360561507167,
        'Median Income ($)': 54888.15721714521,
        'Food Price Index': 105.13451326721136,
        'Inflation Rate (%)': 3.503149309704816,
        'Stock Availability (%)': 74.90832567591417,
        'Transportation Efficiency (%)': 84.92120650799237,
        'Delivery Delays (%)': 10.15981817308801,
        'Storage Capacity (%)': 75.01376609849987,
        'Humidity (%)': humidity,
        'Air Quality Index': air_qua_ind,
        'Heatwave': 0,
        'Cold Wave': 0,
        'Major Event': 0,
        'Tourist Influx (%)': 2.6747972441068657,
        'Fuel Prices': 10.513451326721137,
        'Holiday Indicator': 1 if holiday == "Yes" else 0 ,
        'Year': 2024,
        'Month': 7,
        'Day': 15,
        'Lag_7': 3451.9689574759946,
        'Lag_14': 3452.5349931412898,
        'Lag_30': 3453.0289574759945 ,
        'Rolling_Mean_7': 3451.9493023711543,
        'Rolling_Mean_14': 3452.190010777974,
        'EMA_7': 3451.9633441647557,
        'EMA_14': 3452.1830476131154,
        'Fourier_7': 3453.68401845709,
        'Fourier_14': 3452.7371550279727
    }

    # mapping lattitude with city
    city_latitude_mapping = {
        0: 29.7604,  # City 0
        1: 32.7767,  # City 1
        2: 29.7604,  # City 2
        3: 34.0522,  # City 3
        4: 40.7128,  # City 4
        5: 39.9526,  # City 5
        6: 33.4484,  # City 6
        7: 29.4241,  # City 7
        8: 32.7157,  # City 8
        9: 37.3382 # City 9
    }
    data_dict['Latitude'] = city_latitude_mapping[city]

    # mapping longitude with city
    city_longitude_mapping = {
        0: -87.6298,  # City 0
        1: -96.797,  # City 1
        2: -95.3698,  # City 2
        3:-118.2437,  # City 3
        4: -74.006,  # City 4
        5: -75.1652,  # City 5
        6: -112.074,  # City 6
        7: -98.4936,  # City 7
        8: -117.1611,  # City 8
        9:-121.8863  # City 9
    }
    data_dict['Longitude'] = city_longitude_mapping[city]

    # mapping Population Growth Factor with city
    city_pgf_mapping = {
        0: 1.01,  # City 0
        1: 1.046,  # City 1
        2: 1.034,  # City 2
        3: 1.013,  # City 3
        4: 1.034,  # City 4
        5: 1.016,  # City 5
        6: 1.02,  # City 6
        7: 1.016,  # City 7
        8: 1.018,  # City 8
        9: 1.044  # City 9
    }
    data_dict['Population Growth Factor'] = city_pgf_mapping[city]

    # Crete Data Frame
    df = pd.DataFrame([data_dict])

    # scale data
    X_test_scaled = scaler_X.transform(df)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)[0]
    required_food=y_pred-avail
    required_stat=""
    if(required_food>=0):
        required_stat+="Required Food in City: "+str(int(required_food))+" units"
    else:
        required_stat+="Already Stock Available ✅"
    food_avail_in_food_hub=10345


    path_info,path_img = animate_shortest_path(city)
        
    return f"Food Available in Food Hub: {int(food_avail_in_food_hub)} units\nPredicted Food Demand: {int(y_pred)} units\nAvailable Food in City: {int(avail)} units\n"+required_stat,path_info,path_img


# In[18]:


def plot_transportation_network():
    # Load the transportation cost matrix
    cost_df = pd.read_csv("data/transportation_cost_matrix.csv", index_col=0)

    # Create a graph
    G = nx.Graph()

    # Add nodes (cities + Food Hub)
    for city in cost_df.index:
        G.add_node(city)

    # Add edges with transportation costs, ignoring -1 values (no direct path)
    for i in cost_df.index:
        for j in cost_df.columns:
            if i != j and cost_df.loc[i, j] != -1:
                G.add_edge(i, j, weight=cost_df.loc[i, j])

    # Get positions for nodes
    pos = nx.spring_layout(G, seed=34)

    # Node colors: "Food Hub" is red, others are blue
    node_colors = ["red" if city == "Food Hub" else "#00FFFF" for city in G.nodes]

    # Determine edge colors and thickness based on cost
    edge_colors = []
    edge_weights = []
    min_cost = min(d['weight'] for _, _, d in G.edges(data=True))
    max_cost = max(d['weight'] for _, _, d in G.edges(data=True))

    for u, v, d in G.edges(data=True):
        cost = d['weight']
        # Normalize cost to a 0-1 range
        norm_cost = (cost - min_cost) / (max_cost - min_cost + 1e-6)
        edge_colors.append((norm_cost, norm_cost, 10))  # Green for low cost, red for high
        edge_weights.append(1 + 4 * norm_cost)  # Thicker edges for higher cost

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(12, 8))

    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=3000, edgecolors="black", ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color=edge_colors, width=edge_weights, alpha=0.75, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', font_color='black', ax=ax)

    # Draw edge labels (costs)
    edge_labels = {(u, v): f"{d['weight']}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color="black", ax=ax)

    ax.set_title("City Transportation Network")

    return fig # Return the figure


# In[19]:


def plot_optimized_graph():
    """Load cost matrix, modify it, create an optimized graph, and return the plot figure."""
    # Load the cost matrix
    cost_matrix = pd.read_csv("data/foodCollection_cost_matrix.csv", index_col=0)
    
    # Modify cost matrix
    food_hub_column = "Food Hub"
    direct_connections = cost_matrix[food_hub_column] > 0
    
    np.random.seed(46)
    places_to_remove = np.random.choice(cost_matrix.index[direct_connections], size=3, replace=False)
    
    cost_matrix.loc[places_to_remove, food_hub_column] = -1
    cost_matrix.loc[food_hub_column, places_to_remove] = -1
    
    for place in places_to_remove:
        alternatives = cost_matrix.index[direct_connections & (cost_matrix.index != place)]
        if not alternatives.empty:
            alternative = np.random.choice(alternatives)
            avg_distance = (cost_matrix.loc[place, alternative] + cost_matrix.loc[alternative, food_hub_column]) / 2
            cost_matrix.loc[place, alternative] = avg_distance
            cost_matrix.loc[alternative, place] = avg_distance
    
    # Create a graph
    G = nx.Graph()
    
    # Add nodes
    for place in cost_matrix.index:
        G.add_node(place)
    
    # Add edges
    for i in cost_matrix.index:
        for j in cost_matrix.columns:
            if i != j and cost_matrix.loc[i, j] > 0:
                G.add_edge(i, j, weight=cost_matrix.loc[i, j])
    
    # Get positions for nodes
    pos = nx.spring_layout(G, seed=34)
    
    # Node colors: "Food Hub" is red, others are blue
    node_colors = ["red" if place == "Food Hub" else "#00FFFF" for place in G.nodes]
    
    # Determine edge colors and thickness based on cost
    edge_colors = []
    edge_weights = []
    min_cost = min(d['weight'] for _, _, d in G.edges(data=True))
    max_cost = max(d['weight'] for _, _, d in G.edges(data=True))
    
    for u, v, d in G.edges(data=True):
        cost = d['weight']
        norm_cost = (cost - min_cost) / (max_cost - min_cost + 1e-6)
        edge_colors.append((norm_cost, norm_cost, 10))  # Green for low cost, red for high
        edge_weights.append(1 + 4 * norm_cost)  # Thicker edges for higher cost
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=3000, edgecolors="black", ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color=edge_colors, width=1, alpha=0.75, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', font_color='black', ax=ax)
    
    # Draw edge labels (costs)
    edge_labels = {(u, v): f"{d['weight']:.1f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color="black", ax=ax)
    
    ax.set_title("Food Collection Network")
    
    return fig


# In[20]:


# Function to evaluate the regression model and generate visualizations
def test_xgboost_regression(file):
    try:
        df = pd.read_csv(file.name)
        X = df.drop(columns=['Food Demand'])
        y_true = df['Food Demand'].values.reshape(-1, 1)

        X_test_scaled = scaler_X.transform(X)
        y_pred = model.predict(X_test_scaled)

        # Calculate Metrics
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        accuracy = (1 - (mae / np.mean(y_true))) * 100

        # 📊 Metrics Text Output
        result_text = f"""
        🔍 **Regression Metrics:**  
        ✅ **Mean Absolute Error (MAE):** {mae:.4f}  
        ✅ **Mean Squared Error (MSE):** {mse:.4f}  
        ✅ **Root Mean Squared Error (RMSE):** {rmse:.4f}  
        ✅ **R² Score:** {r2:.4f}  
        ✅ **Accuracy:** {accuracy:.2f}%  
        """

        # 📍 Scatter Plot (Actual vs Predicted)
        fig1, ax1 = plt.subplots()
        ax1.scatter(y_true, y_pred, alpha=0.6, color='blue', edgecolors='k')
        ax1.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--')
        ax1.set_xlabel("Actual Demand")
        ax1.set_ylabel("Predicted Demand")
        ax1.set_title("Actual vs Predicted")
        ax1.grid(True)

        # # 2️⃣ Residual Plot
        y_pred_df=pd.DataFrame(y_pred)
        residuals = y_true - y_pred_df
        
        fig2, ax2 = plt.subplots()
        ax2.scatter(y_pred_df, residuals, color='purple', alpha=0.6)
        ax2.axhline(y=0, color='red', linestyle='--')
        ax2.set_xlabel("Predicted Food Demand")
        ax2.set_ylabel("Residuals (Errors)")
        ax2.set_title("Residual Plot")
        ax2.grid(True)

        # 📍 Histogram of Residuals
        fig3, ax3 = plt.subplots()
        sns.histplot(residuals, bins=30, kde=True, color='orange', ax=ax3)
        ax3.set_xlabel("Residuals")
        ax3.set_title("Histogram of Residuals")

        # 📍 Time Series Plot (Actual vs Predicted)
        fig4, ax4 = plt.subplots()
        ax4.plot(y_true, label="Actual", color="blue")
        ax4.plot(y_pred, label="Predicted", color="red", linestyle="--")
        ax4.set_xlabel("Time")
        ax4.set_ylabel("Demand")
        ax4.set_title("Time Series Plot")
        ax4.legend()

        # 📍 Performance Metrics Bar Chart
        fig5, ax5 = plt.subplots()
        metrics = ["MAE", "RMSE", "R² Score", "Accuracy"]
        values = [mae, rmse, r2, accuracy]
        sns.barplot(x=metrics, y=values, ax=ax5, palette="coolwarm")
        for i, v in enumerate(values):
            ax5.text(i, v + 0.01, f"{v:.4f}", ha="center", fontsize=12, fontweight="bold")
        ax5.set_title("Regression Metrics")

        # 📍 Density Plot (Actual vs Predicted)
        fig6, ax6 = plt.subplots()
        sns.kdeplot(y_true.ravel(), label="Actual", color="blue", shade=True, ax=ax6)
        sns.kdeplot(y_pred.ravel(), label="Predicted", color="red", shade=True, ax=ax6)
        ax6.set_title("Density Plot")
        ax6.legend()

        
        cost_df = pd.read_csv("data/transportation_cost_matrix.csv", index_col=0)

        # 📍 Cost Matrixt
        fig7, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cost_df, annot=True, cmap="coolwarm", fmt="d", ax=ax)
        ax.set_title("Transportation Cost Matrix")

        
        # 📍 City Transportation Network
        fig8=plot_transportation_network()
        fig9=plot_optimized_graph()
        
        return result_text,fig1,fig2,fig3,fig4,fig5,fig6,fig7,fig8,fig9

    except Exception as e:
        return f"❌ Error: {str(e)}", None, None, None, None, None, None, None, None, None


# In[21]:


with gr.Blocks(theme="soft") as app:
    gr.Markdown("# 📊 **Food Demand Prediction Model**")
    with gr.Tabs():
        with gr.Tab("Predict Food Demand In A City"):
            gr.Markdown("Fill Conditions About City **Model Predict Food Demand In The City And Suggest Shortest Path From Food Hub to City**.")
            with gr.Row():
                with gr.Column(scale=1):
                    inputs = [
                    gr.Dropdown(choices=list(city_mapping.values()), label="Select City", value="Jaipur"),
                    gr.Slider(minimum=-4.4, maximum=41.3, step=0.1, label="Temperature (°C)", value=18.4),
                    gr.Slider(minimum=-17.0 , maximum=25.0, step=0.1, label="Rainfall (mm)", value=4.0),
                    gr.Slider(minimum=0, maximum=6, step=1, label="Day of Week (0=Monday, 6=Sunday)", value=0),
                    gr.Slider(minimum=20.0, maximum=89.9, step=0.1, label="Humidity (%)", value=54.9),
                    gr.Slider(minimum=50.0, maximum=199.9, step=0.1, label="Air Quality Index", value=124.9),
                    gr.Radio(choices=["Yes", "No"], label="Is it a Holiday?", value="No"),
                    gr.Number(label="Available Food Quantity in city", precision=2)
                    ]
                    run_button = gr.Button("🔍 Predict Food In The City ")
                with gr.Column(scale=2):
                    outputs = [ gr.Textbox(label="Predicted Food Demand in City"),gr.Textbox(label="Optimal Path Information"), gr.Image(label="Shortest Path Visulization")]
           
            run_button.click(fn=predict_food_demand, inputs=inputs, outputs=outputs)

        with gr.Tab("Model Testing"):
            gr.Markdown("Upload your test dataset and view **model predictions, residuals, and accuracy metrics**.")
            
            with gr.Row():
                with gr.Column(scale=1):
                    input_file = gr.File(label="📂 Upload test CSV File")
                    run_button = gr.Button("🔍 Test Model", variant="primary")
                
                with gr.Column(scale=2):
                    with gr.Group():  # Groups the elements together in a clean section
                        gr.Markdown("### 📊 **Model Performance Summary**", elem_id="summary_title")
                        output_text = gr.Markdown(
                            value="Upload a test file and click **Test Model** to see results.",
                            elem_id="output_text",
                        )
            with gr.Tabs():
                with gr.Tab("📍 Metrics"):
                    with gr.Row():
                        bar_chart = gr.Plot(label="📍 Performance Metrics")
                
                with gr.Tab("📍 Visualizations"):
                    with gr.Row():
                        scatter_plot = gr.Plot(label="📍 Actual vs Predicted Scatter Plot")
                        residual_plot = gr.Plot(label="📍 Residual Plot")
                    with gr.Row():
                        histogram_plot = gr.Plot(label="📍 Histogram of Residuals")
                        time_series_plot = gr.Plot(label="📍 Time Series Plot")
                    with gr.Row():
                        density_plot = gr.Plot(label="📍 Density Plot")
                        cost_matrix = gr.Plot(label="📍 Cost Matrix(No Direct Path -> -1)")
                    with gr.Row():
                        transportation_network = gr.Plot(label="📍 City Transportation Network")
                        dm = gr.Plot(label="📍 Food Collection Network")
        
            run_button.click(
               fn=test_xgboost_regression,
                inputs=[input_file],
                outputs=[output_text, scatter_plot, residual_plot, histogram_plot, time_series_plot, bar_chart, density_plot, cost_matrix, transportation_network, dm],
            )


# In[22]:


app.launch(share=True)


# In[ ]:




