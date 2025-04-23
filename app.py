#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import numpy as np
import csv
import os
import seaborn as sns
import gradio as gr
from PIL import Image
import joblib
import xgboost as xgb
import pickle  # To load the saved model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import plotly.graph_objects as go
import time
from datetime import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.message import EmailMessage
from email.mime.image import MIMEImage
from email.utils import formataddr
import re
import plotly.express as px
from newsapi import NewsApiClient
import datetime as dt
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings


# In[18]:


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


# In[19]:


# Load pre-trained XGBoost regression model
with open("models/xgb_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load the feature scaler
scaler_X = joblib.load("models/scaler_X.pkl")


# In[20]:


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


# In[21]:


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


# In[22]:


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
    food_avail_in_food_hub=10345 #for example


    path_info,path_img = animate_shortest_path(city)
        
    return f"Food Available in Food Hub: {int(food_avail_in_food_hub)} units\nPredicted Food Demand: {int(y_pred)} units\nAvailable Food in City: {int(avail)} units\n"+required_stat,path_info,path_img


# In[23]:


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


# In[24]:


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
    
    ax.set_title("Optimized Food Distribution Network")
    
    return fig


# In[25]:


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


# In[26]:


# --------------Food donation-------------


# Distance from central hub
hub_distances = {
    "Delhi": 1934.0, "Mumbai": 305.2, "Patna": 360.0, "Jaipur": 2039.0,
    "Kolkata": 1907.0, "Chandigarh": 650.0, "Pune": 397.0,
    "Bangalore": 631.0, "Lucknow": 1647.0, "Agra":684.0
}

# CSV file to save donations
csv_file = "data/donations_food.csv"
try:
    with open(csv_file, "x") as f:
        f.write("Name,Phone,Email,City,Food_Quantity,Date,Distance_From_Hub,Message\n")
except FileExistsError:
    pass


    
def send_thank_you_email(to_email, donor_name, food_quantity, city):
    EMAIL_ADDRESS = "kausar.project2025@gmail.com"
    EMAIL_PASSWORD = "pafq ijrp djdf ehpd"

    # Create the base message
    msg = MIMEMultipart('related')
    msg['Subject'] = "Thanks for Your Food Donation! 💚"
    msg['From'] = formataddr(("Food Donation Team", EMAIL_ADDRESS))
    msg['To'] = to_email

    # Create the alternative part for plain text and HTML
    alt_part = MIMEMultipart('alternative')

    # Plain text version
    plain_text = f"""\
Hi {donor_name},

Thank you for donating {food_quantity} units of food from {city}! 💚
Your generosity is deeply appreciated, and a pickup volunteer will contact you soon.

Together, we fight food waste and hunger. 🙌

- The Food Donation Team
"""
    alt_part.attach(MIMEText(plain_text, 'plain'))

    # HTML version with image inline
    html_content = f"""\
<html>
  <body>
    <p>Hi <b>{donor_name}</b>,</p>
    <p>Thank you for donating <b>{food_quantity}</b> units of food from <b>{city}</b>! 💚</p>
    <p>Your generosity is deeply appreciated, and a pickup volunteer will contact you soon.</p>
    <p>Together, we fight food waste and hunger. 🙌</p>
    <img src="cid:thanksimg" alt="Thank You Image" style="margin-top:15px; max-width:400px;">
    <p>— The Food Donation Team</p>
  </body>
</html>
"""
    alt_part.attach(MIMEText(html_content, 'html'))

    # Attach the alternative part to the main message
    msg.attach(alt_part)

    # Add the image
    with open("data/thanks.png", 'rb') as img:
        img_data = img.read()
        img_attachment = MIMEImage(img_data)
        img_attachment.add_header('Content-ID', '<thanksimg>')
        img_attachment.add_header('Content-Disposition', 'inline', filename='thanks.png')
        msg.attach(img_attachment)

    # Send the message
    with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
        smtp.starttls()
        smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        smtp.send_message(msg)


def is_valid_email(email):
    # Basic regex for email validation
    pattern = r"[^@]+@[^@]+\.[^@]+"
    return re.match(pattern, email) is not None


# Donation Function
def donate_food(username, phone, email, city, food_quantity, message):
    if username.strip() == "" or email.strip() == "":
        return "❗ Please fill out your name and email.", None
    if not is_valid_email(email):
        return "❗ Please enter a valid email address.", None

    try:
        quantity = float(food_quantity)
        if quantity <= 0:
            return "❗ Quantity must be a positive number.", None
    except ValueError:
        return "❗ Please enter a valid number for quantity.", None

    date = datetime.now().strftime("%Y-%m-%d")
    distance = hub_distances.get(city, 0.0)
    pickup_hour = min(int(4 + distance * 0.1), 8)
    pickup_time = f"{pickup_hour}:00 PM"

    # Save to CSV
    df = pd.DataFrame([[username, phone, email, city, food_quantity, date, distance, message]],
                      columns=["Name", "Phone", "Email", "City", "Food_Quantity", "Date", "Distance_From_Hub", "Message"])
    df.to_csv(csv_file, mode='a', header=False, index=False)

    # Send Thank You Email
    send_thank_you_email(email, username, food_quantity, city)

    msg = f"""✅ **Thank you, {username}!👏💚**
You’ve donated **{quantity} units** from **{city}**.
Pickup man will shortly contact you.
"""
    image_path = "data/thanks.png"
    return msg, image_path


# In[27]:


# ---------Aunthentication--------

CSV_PATH = "data/users.csv"
# Ensure CSV exists
if not os.path.exists(CSV_PATH):
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    with open(CSV_PATH, "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["USER", "Password", "Email", "Contact", "Address", "Approved"])
# Load users from CSV
def load_users_for_auth():
    users = {}
    with open(CSV_PATH, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            users[row['USER'].strip()] = {
                'password': row['Password'],
                'email': row['Email'],
                'contact': row['Contact'],
                'address': row['Address'],
                'approved': row['Approved'].strip().lower() == 'true'
            }
    return users


# Login logic
def login_user(username, password):
    users = load_users_for_auth()

    if not username or not password:
        return (
            "❗ Please enter both username and password.",
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False)
        )

    if username in users:
        if users[username]['password'] == password:
            if username.lower() == "admin":
                return (
                    "👑 Welcome Admin!",
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=True)
                )
            if not users[username]['approved']:
                return (
                    "⏳ Your account is pending admin approval.",
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=False)
                )
            else:
                return (
                    f"✅ Welcome, {username}!",
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False)
                )
        else:
            return (
                "❌ Incorrect password.",
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False)
            )
    else:
        return (
            "⚠️ User does not exist. Please register.",
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False)
        )

# Register logic
def register_user(username, password, email, contact, address):
    users = load_users_for_auth()
    if not all([username, password, email, contact, address]):
        return "❗ Please fill in all fields."

    if username in users:
        return "⚠️ This username is already taken."
    if not is_valid_email(email):
        return "❗ Please enter a valid email address."
 

    with open(CSV_PATH, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([username, password, email, contact, address, "false"])

    return "🆕 Registered successfully! ✅ Now please wait for admin approval and login."


# In[28]:


# ------Admin Panel--------


USER_CSV = "data/users.csv"

# 🔍 Load users from CSV
def load_users():
    USER_CSV = "data/users.csv"
    with open(USER_CSV, newline='') as csvfile:
        return list(csv.DictReader(csvfile))


def get_user_stats():
    users = load_users()
    total = len(users)
    # approved = sum(1 for user in users.values() if str(user.get("approved", False)).lower() == "true")
    approved = sum(1 for user in users if str(user.get("Approved", "")).lower() == "true")

    pending = total - approved
    return total, approved, pending


# 📋 Get pending usernames
def get_pending_users():
    return [u["USER"] for u in load_users() if str(u.get("Approved", "")).lower() != "true"]


# 📋 Get all usernames
# def get_all_usernames():
#     return [u["USER"] for u in load_users()]

def get_all_usernames():
    # return list(load_users().keys())
    return [u["USER"] for u in load_users()]





def refresh_stats():
    pending_users_list=get_pending_users()
    total, approved, pending = get_user_stats()
    return (
        f"<div style='font-size: 30px;'>👥 <strong>Total Users:</strong> {total}</div>",
        f"<div style='font-size: 30px; color: green;'>✅ <strong>Approved Users:</strong> {approved}</div>",
        f"<div style='font-size: 30px; color: orange;'>⏳ <strong>Pending Users:</strong> {pending}</div>",
        draw_pie_chart(),
        pending_users_list
    )

# ✅ Approve a user
def approve_user(username):
    users = load_users()
    found = False
    email = None

    for user in users:
        if user["USER"] == username:
            user["Approved"] = "true"
            found = True
            email = user.get("Email", "")
            break

    if found:
        with open(USER_CSV, 'w', newline='') as csvfile:
            fieldnames = users[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(users)

        # ✅ Send email after saving to CSV
        if email:
            send_approval_email(email, username)

        return (
            f"✅ Approved {username}!",
            gr.update(choices=get_pending_users()),
            draw_pie_chart(),
            gr.update(choices=get_all_usernames())
        )
    else:
        return (
            "❌ User not found",
            gr.update(choices=get_pending_users()),
            draw_pie_chart(),
            gr.update(choices=get_all_usernames())
        )



# ❌ Reject a user
def reject_user(username):
    users = load_users()
    new_users = [user for user in users if user["USER"] != username]

    if len(new_users) < len(users):
        with open(USER_CSV, 'w', newline='') as csvfile:
            fieldnames = new_users[0].keys() if new_users else users[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(new_users)
        return (
            f"❌ Rejected {username} (User removed)",
            gr.update(choices=get_pending_users()),
            draw_pie_chart(),
            gr.update(choices=get_all_usernames())
        )
    else:
        return (
            "⚠️ User not found",
            gr.update(choices=get_pending_users()),
            draw_pie_chart(),
            gr.update(choices=get_all_usernames())
        )

# 🥧 Draw Pie Chart
def draw_pie_chart():
    _, approved, pending = get_user_stats()
    fig = go.Figure(data=[go.Pie(
        labels=["Approved", "Pending"],
        values=[approved, pending],
        hole=0.4,
        marker=dict(colors=["#27ae60", "#e67e22"]),
    )])
    fig.update_layout(title_text="User Approval Status")
    return fig

# 👤 Get formatted user details
def get_user_details(username):
    users = load_users()
    for user in users:
        if user["USER"] == username:
            approved = user.get("Approved", "N/A").lower()
            status_badge = (
                "🟢 <span style='color:limegreen;font-weight:bold'>Approved</span>" if approved == "true"
                else "🟠 <span style='color:orange;font-weight:bold'>Pending</span>"
            )

            return f"""
<div style='padding: 16px; border: 1px solid #444; border-radius: 12px; background-color: #f0f0f0; color: #2c3e50; font-family: Arial, sans-serif;'>
    <h3 style='margin-top: 0;'>👤 <span style="color:#2c3e50;">{user['USER']}</span></h3>
    <p>📧 <strong style="color:#2c3e50;">Email:</strong> <span style="color:#2c3e50;">{user.get('Email', 'N/A')}</span></p>
    <p>📞 <strong style="color:#2c3e50;">Contact:</strong> <span style="color:#2c3e50;">{user.get('Contact', 'N/A')}</span></p>
    <p>🏠 <strong style="color:#2c3e50;">Address:</strong> <span style="color:#2c3e50;">{user.get('Address', 'N/A')}</span></p>
    <p>🔓  <strong style="color:#2c3e50;">Status:</strong> <span>{status_badge}</span></p>
</div>
"""
    return "<p style='color:red;'>⚠️ User not found</p>"



# Delete user logic
def delete_user(username):
    users = load_users()
    updated_users = [u for u in users if u["USER"] != username]

    if len(updated_users) < len(users):
        with open(USER_CSV, 'w', newline='') as csvfile:
            fieldnames = updated_users[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(updated_users)
        return (
            f"🗑️ Deleted user: {username}",
            gr.update(choices=get_all_usernames()),
            gr.update(choices=get_pending_users()),
            draw_pie_chart()
        )
    else:
        return (
            "⚠️ User not found.",
            gr.update(choices=get_all_usernames()),
            gr.update(choices=get_pending_users()),
            draw_pie_chart()
        )


#Approval email sendinding logic

EMAIL_ADDRESS = os.getenv("kausar.project2025@gmail.com")
EMAIL_PASSWORD = os.getenv("pafq ijrp djdf ehpd")

def is_valid_email(email):
    # Basic regex for email validation
    pattern = r"[^@]+@[^@]+\.[^@]+"
    return re.match(pattern, email) is not None

def send_approval_email(to_email, username):
    msg = EmailMessage()
    msg['Subject'] = "🎉 Your Account Has Been Approved!"
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = to_email
    msg.set_content(f"Hi {username},\n\nYour account has been approved by the admin. You can now access the platform.\n\nThanks!")

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
        return True
    except Exception as e:
        print("❌ Email sending failed:", e)
        return False


# Adding new user logic

def add_new_user(username, password, email, contact, address, approved):
    if not username or not password:
        return "⚠️ Username and Password are required."
    if not is_valid_email(email):
        return "❗ Please enter a valid email address."
    
    users = load_users()
    if any(u["USER"] == username for u in users):
        return "❌ Username already exists."
    
    new_user = {
        "USER": username,
        "Password": password,
        "Email": email,
        "Contact": contact,
        "Address": address,
        "Approved": "true" if approved=="Yes" else "false"
    }
    users.append(new_user)
    
    with open(USER_CSV, 'w', newline='') as csvfile:
        fieldnames = new_user.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(users)
    
    return f"✅ User {username} added successfully!"



# User updation logi
def load_user_data(username):
    users = load_users()
    for user in users:
        if user["USER"] == username:
            return (
                user.get("Email", ""),
                user.get("Contact", ""),
                user.get("Address", ""),
                user.get("Approved", "false")
            )
    return "", "", "", "false"
    

# Update logic
def update_user_info(username, email, contact, address, approved):
    users = load_users()
    updated = False
    for user in users:
        if user["USER"] == username:
            user["Email"] = email
            user["Contact"] = contact
            user["Address"] = address
            user["Approved"] = "true" if approved=="Yes" else "false"
            updated = True
            break
    
    if updated:
        with open(USER_CSV, 'w', newline='') as csvfile:
            fieldnames = users[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(users)
        return f"✅ User '{username}' updated successfully."
    else:
        return "⚠️ User not found."


# logic for displaying food donation details
# 📦 Load and show recent 5 food donations
import pandas as pd

DONATION_CSV = "data/donations_food.csv"  # adjust path as needed

def load_recent_donations():
    df = pd.read_csv(DONATION_CSV)
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    recent = df.tail(5)

    html = """
    <div style='font-family: Arial; padding: 10px;'>
        <h4 style="margin-bottom:10px;">📦 Recent Food Donations</h4>
        <table style="width:100%; border-collapse: collapse;">
            <thead>
                <tr style="background-color: #2c3e50; color: white;">
                    <th style="padding: 8px; border: 1px solid #ccc;">👤 Name</th>
                    <th style="padding: 8px; border: 1px solid #ccc;">📧 Email</th>
                    <th style="padding: 8px; border: 1px solid #ccc;">📞 Phone</th>
                    <th style="padding: 8px; border: 1px solid #ccc;">🏙️ City</th>
                    <th style="padding: 8px; border: 1px solid #ccc;">🍱 Quantity</th>
                    <th style="padding: 8px; border: 1px solid #ccc;">📅 Date</th>
                    <th style="padding: 8px; border: 1px solid #ccc;">🗺️ Distance</th>
                    <th style="padding: 8px; border: 1px solid #ccc;">✉️ Message</th>
                </tr>
            </thead>
            <tbody>
    """

    for _, row in recent.iterrows():
        html += f"""
            <tr>
                <td style="padding: 8px; border: 1px solid #ccc;">{row['Name']}</td>
                <td style="padding: 8px; border: 1px solid #ccc;">{row['Email']}</td>
                <td style="padding: 8px; border: 1px solid #ccc;">{row['Phone']}</td>
                <td style="padding: 8px; border: 1px solid #ccc;">{row['City']}</td>
                <td style="padding: 8px; border: 1px solid #ccc;">{row['Food_Quantity']}</td>
                <td style="padding: 8px; border: 1px solid #ccc;">{row['Date'].strftime('%Y-%m-%d')}</td>
                <td style="padding: 8px; border: 1px solid #ccc;">{row['Distance_From_Hub']}</td>
                <td style="padding: 8px; border: 1px solid #ccc;">{row.get('Message', 'N/A')}</td>
            </tr>
        """

    html += "</tbody></table></div>"
    return html




# In[29]:


# Dashboard
# Load the CSV data
df = pd.read_csv('data/donations.csv')

# Clean data - remove duplicates
df = df.drop_duplicates()

# Calculate metrics
total_donations = df['food_quantity'].sum()
unique_cities = df['city'].nunique()
unique_donors = df['username'].nunique()
avg_distance = df['distance_from_hub'].mean()

# Function to create city-wise donations chart
def create_city_donations_chart():
    city_donations = df.groupby('city')['food_quantity'].sum().reset_index()
    city_donations = city_donations.sort_values(by='food_quantity', ascending=False)
    
    fig = px.bar(
        city_donations, 
        x='city', 
        y='food_quantity',
        color='food_quantity',
        color_continuous_scale='Viridis',
        title='Food Donations by City',
        labels={'food_quantity': 'Food Quantity (kg)', 'city': 'City'}
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#333333'),
        margin=dict(l=20, r=20, t=40, b=20),
        coloraxis_showscale=False
    )
    return fig

# Function to create donor distribution pie chart
def create_donor_distribution():
    donor_counts = df.groupby('username').size().reset_index(name='count')
    donor_counts = donor_counts.sort_values(by='count', ascending=False)
    
    # Take top 5 donors and group others
    if len(donor_counts) > 5:
        top_donors = donor_counts.iloc[:5]
        others_count = donor_counts.iloc[5:]['count'].sum()
        top_donors = pd.concat([top_donors, pd.DataFrame([['Others', others_count]], columns=['username', 'count'])])
        donor_counts = top_donors
    
    fig = px.pie(
        donor_counts, 
        values='count', 
        names='username',
        title='Top Donors Distribution',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#333333'),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

# Function to create distance vs quantity scatter plot
def create_distance_quantity_scatter():
    fig = px.scatter(
        df, 
        x='distance_from_hub', 
        y='food_quantity',
        color='city',
        title='Food Quantity vs Distance',
        labels={'food_quantity': 'Food Quantity (kg)', 'distance_from_hub': 'Distance from Hub (km)'},
        hover_data=['username', 'city']
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#333333'),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

# Function to create a gauge chart for monthly target
def create_gauge_chart():
    monthly_target = 100  # Example target
    current_progress = min(total_donations / monthly_target * 100, 100)
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = current_progress,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Monthly Target Progress (%)"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "#2E8B57"},
            'steps': [
                {'range': [0, 50], 'color': "#FFEBEE"},
                {'range': [50, 75], 'color': "#FFCCBC"},
                {'range': [75, 100], 'color': "#C8E6C9"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#333333'),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

# Create a table of recent donations
def create_recent_donations_table():
    recent_df = df.sort_values(by='date', ascending=False).head(5)
    recent_df = recent_df[['username', 'city', 'food_quantity', 'date']]
    return recent_df.rename(columns={
        'username': 'Donor', 
        'city': 'City', 
        'food_quantity': 'Amount (kg)', 
        'date': 'Date'
    })

# Define the upcoming cities (from your example)
upcoming_cities = ["Gurgaon", "Ludhiana", "Chennai", "Ranchi", "Surat", "Hyderabad", "Patna", "Lucknow", "Nagpur", "Kanpur"]

# Define About Us content
about_us_content = """
# About Food Donation Network

Our mission is to connect food donors with people in need to reduce food waste and fight hunger. 
We operate in multiple cities across India, building a network of donors, volunteers, and beneficiaries.

## Our Impact

- **12,345+** food units donated
- **10** cities served
- **4** teams deployed
- **100+** families supported monthly

## How It Works

1. **Donate**: Register as a donor and list available food
2. **Connect**: We match your donation with nearby recipients
3. **Deliver**: Our volunteers ensure safe and timely delivery

Join our mission to build a hunger-free society!
"""


# In[30]:


# News Section

# ✅ Your NewsAPI key here
NEWSAPI_KEY = "0d64617d281a40abb4bf15ea7b47c577"
newsapi = NewsApiClient(api_key=NEWSAPI_KEY)

# Keywords for relevant news
search_keywords = ["food crisis", "food shortage", "disaster", "drought", "hunger", "emergency aid"]

def get_food_crisis_news():
    query = " OR ".join(search_keywords)
    try:
        top_headlines = newsapi.get_everything(q=query,
                                               language='en',
                                               sort_by='publishedAt',
                                               page_size=5)
        articles = top_headlines['articles']
        if not articles:
            return "🚨 No urgent food-related news found right now."

        result = ""
        for a in articles:
            published = dt.datetime.strptime(a['publishedAt'], "%Y-%m-%dT%H:%M:%SZ")
            result += f"""
<div class='news-item'>
  <strong>{a['title']}</strong><br>
  📅 {published.strftime('%b %d, %Y %H:%M')}<br>
  🌍 Source: {a['source']['name']}<br>
  <a href="{a['url']}" target="_blank" class='news-link'>🔗 Read Full Article</a>
</div>
"""
        return result
    except Exception as e:
        return f"❌ Error fetching news: {e}"


# In[31]:


# Gradio Interface

with gr.Blocks(gr.themes.Ocean(),css="""
.news-item {
  display: block;
  padding: 10px;
  border-bottom: 1px solid #ccc;
  margin-bottom: 8px;
  font-size: 15px;
}
.news-link {
  background-color: #1f6feb;
  color: white;
  text-decoration: none;
  padding: 6px 12px;
  border-radius: 6px;
  display: inline-block;
  margin-top: 6px;
}
.news-link:hover {
  background-color: #174ea6;
}
""") as app:
    
    gr.Markdown("# 📊 **Food Demand Prediction And Donation Plateform**")
    with gr.Column(visible=False) as app_section:
        with gr.Row():
            gr.Markdown("**Food Demand Predictor:** Accurately forecasts food quantity needs using real-time data and AI models.")
        with gr.Row():
            gr.Markdown("")
        with gr.Row():
            gr.Markdown("")
        with gr.Row():
            gr.Markdown("")
        with gr.Row():
            gr.Markdown("")
    
        with gr.Row():
            gr.Image(value="data/1.jpeg", label="Image 1", show_label=False)
            gr.Image(value="data/3.jpeg", label="Image 2", show_label=False)
            gr.Image(value="data/2.jpeg", label="Image 3", show_label=False)
    
    
        with gr.Row():
            gr.Markdown("")
        with gr.Row():
            gr.Markdown("")
        with gr.Row():
            gr.Markdown("")
        with gr.Row():
            gr.Markdown("")
        with gr.Row():
            gr.Markdown("")
    
        with gr.Row():
            gr.Markdown("The **Food Demand Prediction Model** is an innovative machine learning-driven solution designed to optimize and enhance the process of food donation in underserved communities. Byleveraging data analytics, predictive modeling, and real-time information, this project aims to reduce food waste while increasing the efficiency and reach of food donations to areas in need.")
        with gr.Row():
            gr.Markdown("")
        with gr.Row():
            gr.Markdown("")
        with gr.Row():
            gr.Markdown("")
        # with gr.Row():
        #     gr.Markdown("**Smart Giving, Zero Waste, Full Plates! 🍽️🥞🍱**")
        with gr.Row():
            with gr.Column():
                gr.Markdown("")
            with gr.Column():
                gr.Markdown("**Smart Giving, Zero Waste, Full Plates! 🍽️🥞🍱**")
            with gr.Column():
                gr.Markdown("")
        
        with gr.Row():
            gr.Markdown("")
        with gr.Row():
            gr.Markdown("")
        with gr.Row():
            gr.Markdown("")

        # Food Demand prediction tab
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
                        run_button = gr.Button("🔍 Predict Food In The City ", variant="primary")
                    with gr.Column(scale=2):
                        outputs = [ gr.Textbox(label="Predicted Food Demand in City"),gr.Textbox(label="Optimal Path Information"), gr.Image(label="Shortest Path Visulization")]
               
                run_button.click(fn=predict_food_demand, inputs=inputs, outputs=outputs)
                
            # Food donation tab
            with gr.Tab("Food Donation Portal"):
                gr.Markdown("Help reduce food waste by donating your surplus food!🍱")
                with gr.Row():
                    username = gr.Textbox(label="Your Name", placeholder="Enter your name")
                    phone = gr.Textbox(label="Contact Number", placeholder="e.g. 9876543210")
                with gr.Row():
                    email = gr.Textbox(label="Email", placeholder="Enter your email")
                    city = gr.Dropdown(label="Select City", choices=list(hub_distances.keys()))
                with gr.Row():
                    quantity = gr.Textbox(label="Food Quantity (in units)", placeholder="e.g. 10 or 12.5")
                    message = gr.Textbox(label="Your Message to us ")
                with gr.Row():
                    with gr.Column():
                        donate_btn = gr.Button("Donate Now ✅",variant="primary")
                        output_msg = gr.Markdown()
                        gr.Markdown("### ❤️ What Donors Are Saying:")
                        gr.Markdown("""
                         - “Great initiative! 💚 — Ayesha”
                         - “Happy to help! 🍱 — Rahul”
                         - “Feels good to give back 🙌 — Meera”
                         - “Proud to contribute 👏 — Arjun”
                         - “Donated from my hostel! 😇 — Tanvi”
                        """)
                    with gr.Column():
                        thanks_img=gr.Image(label="Image", show_label=False)

                donate_btn.click(
                    fn=donate_food,
                    inputs=[username,phone, email, city, quantity ,message],
                    outputs=[output_msg,thanks_img]
                )
        
                     

            # dashboard tab
            with gr.Tab("Dashboard"):
                gr.Markdown(
                    """
                    # Food Donation Dashboard
                    A live summary of our impact and upcoming missions.
                    """
                )
                # Top metrics
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("## 🏙️ Cities Served")
                        gr.Markdown(f"### {unique_cities}")
                    with gr.Column():
                        gr.Markdown("## 👥 People Connected")
                        gr.Markdown(f"### {unique_donors}")
                    with gr.Column():
                        gr.Markdown("## 🍲 Food Units Donated")
                        gr.Markdown(f"### {total_donations:.0f} kg")
                    with gr.Column():
                        gr.Markdown("## 📍 Avg. Distance")
                        gr.Markdown(f"### {avg_distance:.1f} km")

                # Main dashboard content
                with gr.Tabs():
                    with gr.TabItem("Dashboard"):
                        with gr.Row():
                            with gr.Column(scale=2):
                                gr.Markdown("### Food Donations by City")
                                city_chart = gr.Plot(value=create_city_donations_chart())
                            with gr.Column(scale=1):
                                gr.Markdown("### Monthly Target")
                                gauge_chart = gr.Plot(value=create_gauge_chart())
                
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("### Donor Distribution")
                                donor_pie = gr.Plot(value=create_donor_distribution())
                            with gr.Column():
                                gr.Markdown("### Food Quantity vs Distance")
                                scatter_plot = gr.Plot(value=create_distance_quantity_scatter())
                
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("### Recent Donations")
                                recent_table = gr.DataFrame(value=create_recent_donations_table())
                            with gr.Column():
                                gr.Markdown("### Upcoming Cities to Connect")
                                gr.Markdown(", ".join(upcoming_cities))


                    with gr.TabItem("Map View"):
                        gr.Markdown("### Donation Locations")
                        # Display a local image instead of fetching from URL
                        gr.Image(value=r"data/map.webp", label="Distribution Map")
                
                        gr.Markdown("### Distribution Hubs")
                        with gr.Row():
                            for city in df['city'].unique()[:4]:
                                with gr.Column():
                                    gr.Markdown(f"#### {city} Hub")
                                    gr.Markdown(f"Total Donations: **{df[df['city'] == city]['food_quantity'].sum():.1f} kg**")
                                    gr.Markdown(f"Active Donors: **{df[df['city'] == city]['username'].nunique()}**")
    

                
          
                        with gr.TabItem("About Us"):
                            with gr.Row():
                                with gr.Column():
                                    gr.Markdown(about_us_content)
                                with gr.Column():
                                    gr.Image(value=r"data/teampic.jpg", label="Our Team")
    
                            gr.Markdown("### Contact Us")
                            with gr.Row():
                                name_input = gr.Textbox(label="Your Name")
                                contact_input = gr.Textbox(label="Contact Number")
                            message_input = gr.Textbox(label="Message to Us", lines=3)
                            submit_btn = gr.Button("Submit")
                
                            submit_btn.click(
                                fn=lambda name, contact, message: gr.Markdown(f"Thank you {name}! We'll get back to you soon."),
                                inputs=[name_input, contact_input, message_input],
                                outputs=gr.Markdown()
                            )

                gr.Markdown(f"© {datetime.now().year} Food Donation Network. Data last updated: {datetime.now().strftime('%Y-%m-%d')}")      
           
            with gr.Tab("Current News"):
                gr.Markdown("## 🌍 Global Food Crisis & Disaster Alerts")
                news_box = gr.HTML(value=get_food_crisis_news(), elem_id="news-output")
                gr.Button("🔄 Refresh News",variant="primary").click(fn=get_food_crisis_news, outputs=news_box)

            # model testing tab
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
                            dm = gr.Plot(label="📍 City Transportation Network")
            
                run_button.click(
                   fn=test_xgboost_regression,
                    inputs=[input_file],
                    outputs=[output_text, scatter_plot, residual_plot, histogram_plot, time_series_plot, bar_chart, density_plot, cost_matrix, transportation_network, dm],
                )
            
    with gr.Column(visible=True) as login_register_section:
        with gr.Tab("Login"):
            with gr.Row():
                with gr.Column(scale=2):
                    login_user_input = gr.Textbox(label="Username")
                    login_pass_input = gr.Textbox(label="Password", type="password")
                    login_btn = gr.Button("Login", variant="primary")
                    login_msg = gr.Textbox(label="", interactive=False)
                with gr.Column(scale=1):
                    # gr.Markdown("hello world")
                    gr.Image(value="data/login.jpeg", label="Image", show_label=False)
        with gr.Tab("Register"):
            with gr.Row():
                with gr.Column(scale=1):
                    reg_user = gr.Textbox(label="Username")
                    reg_pass = gr.Textbox(label="Password", type="password")
                    reg_email = gr.Textbox(label="Email")
                    reg_contact = gr.Textbox(label="Contact Number")
                    reg_address = gr.Textbox(label="Address")
                    register_btn = gr.Button("Register", variant="primary")
                    register_msg = gr.Textbox(label="")
                with gr.Column(scale=1):
                    # gr.Markdown("hello world")
                    gr.Image(value="data/login.jpeg", label="Image", show_label=False)
                
    with gr.Column(visible=False) as admin_panel:
        gr.Markdown("## 🛠️ Admin Panel")
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    gr.Markdown("")
                with gr.Row():
                    gr.Markdown("")
                with gr.Row():
                    gr.Markdown("")
                with gr.Row():
                    gr.Markdown("")
                with gr.Row():
                    gr.Markdown("")
                with gr.Row():
                    gr.Markdown("")
                with gr.Row():
                     # 📈 Stats + Chart
                    total_users = gr.Markdown()
                    approved_users = gr.Markdown()
                    pending_users = gr.Markdown()
                with gr.Row():
                    gr.Markdown("")
                with gr.Row():
                    gr.Markdown("")
                with gr.Row():
                    gr.Markdown("")
                with gr.Row():
                    gr.Markdown("")
    
                with gr.Row():
                    refresh_btn = gr.Button("🔄 Refresh Stats",variant="primary")
            with gr.Column(scale=1):
                pie_chart = gr.Plot(label="User Approval Chart")

    
        # refresh_btn.click(refresh_stats, outputs=[total_users, approved_users, pending_users, pie_chart])
        # app.load(refresh_stats, outputs=[total_users, approved_users, pending_users, pie_chart])



        
        with gr.Row():
            with gr.Column():
                # ✅ Approve / ❌ Reject section
                gr.Markdown("### ⚙️ Approve / Reject Users")
                pending_users_list = gr.Dropdown(choices=get_pending_users(), label="Pending Users")
                with gr.Row():
                    approve_btn = gr.Button("✅ Approve", variant="primary")
                    reject_btn = gr.Button("❌ Reject", variant="primary")
                admin_output = gr.Markdown()
    

                
            with gr.Column():
                # 👁️ View User Details
                # gr.Markdown("---")
                gr.Markdown("### 🔍 View User Details")
                user_detail_dropdown = gr.Dropdown(choices=get_all_usernames(), label="Select User")
                user_detail_output = gr.HTML()
            
                user_detail_dropdown.change(fn=get_user_details, inputs=user_detail_dropdown, outputs=user_detail_output)
                
        refresh_btn.click(refresh_stats, outputs=[total_users, approved_users, pending_users, pie_chart,pending_users_list])
        app.load(refresh_stats, outputs=[total_users, approved_users, pending_users, pie_chart,pending_users_list])

        approve_btn.click(
        fn=approve_user,
        inputs=pending_users_list,
        outputs=[admin_output, pending_users_list, pie_chart, pending_users_list]
        )

        reject_btn.click(
            fn=reject_user,
            inputs=pending_users_list,
            outputs=[admin_output, pending_users_list, pie_chart, pending_users_list]
        )
        
        # ➕ Add New User
        gr.Markdown("### ➕ Add New User")
        with gr.Row():
            new_username = gr.Textbox(label="Username")
            new_password = gr.Textbox(label="Password", type="password")  # <-- added
        with gr.Row():
            new_email = gr.Textbox(label="Email")
            new_contact = gr.Textbox(label="Contact")
        with gr.Row():
            new_address = gr.Textbox(label="Address")
            new_approved = gr.Radio(choices=["Yes", "No"], label="Approved?", value="No")
    
        with gr.Row():
            with gr.Column():
                add_user_btn = gr.Button("➕ Add User", variant="primary")
            with gr.Column():
                add_user_output = gr.Markdown()   
    
                
        add_user_btn.click(
            fn=add_new_user,
            inputs=[new_username, new_password, new_email, new_contact, new_address, new_approved],
            outputs=add_user_output
        )

        # Update user details
        gr.Markdown("---")
        gr.Markdown("### ✏️ Update User Information")  
        # Dropdown to select user
        update_user_dropdown = gr.Dropdown(choices=get_all_usernames(), label="Select User to Update")
    
        # Editable input fields
        update_email = gr.Textbox(label="Email")
        update_contact = gr.Textbox(label="Contact")
        update_address = gr.Textbox(label="Address")
        update_status = gr.Radio(choices=["Yes", "No"], label="Approved", value="No")
        
        with gr.Row():
            with gr.Column():
                update_btn = gr.Button("💾 Update User", variant="primary")
        with gr.Row():
            with gr.Column():
                update_message = gr.Markdown()
                
        update_user_dropdown.change(
        load_user_data,
        inputs=update_user_dropdown,
        outputs=[update_email, update_contact, update_address, update_status]
        )
        
        update_btn.click(
            update_user_info,
            inputs=[update_user_dropdown, update_email, update_contact, update_address, update_status],
            outputs=update_message
        )

        # Delete input fields
        gr.Markdown("### 🗑️ Delete Any User")
        
        delete_user_dropdown = gr.Dropdown(choices=get_all_usernames(), label="Select User to Delete")
    
        delete_user_btn = gr.Button("Delete User", variant="primary")

        delete_user_output = gr.Textbox(label="", interactive=False)
        
        delete_user_btn.click(
            fn=delete_user,
            inputs=delete_user_dropdown,
            outputs=[delete_user_output, user_detail_dropdown, pending_users_list, pie_chart]
        )
        gr.Markdown("---")
        gr.Markdown("### 🥗 Recent Food Donations (Latest 5)")
        
        donation_display = gr.HTML()
        donation_refresh = gr.Button("🔄 Refresh Donations",variant="primary")
        
        donation_refresh.click(fn=load_recent_donations, outputs=donation_display)
        app.load(load_recent_donations, outputs=donation_display)


    login_btn.click(
        fn=login_user,
        inputs=[login_user_input, login_pass_input],
        outputs=[login_msg, app_section, login_register_section, admin_panel]
    )

    register_btn.click(
        fn=register_user,
        inputs=[reg_user, reg_pass, reg_email, reg_contact, reg_address],
        outputs=register_msg
    )


# In[32]:


app.launch()


# In[ ]:





# In[ ]:




