import streamlit as st
import pandas as pd
 
# import matplotlib.pyplot as plt
import plotly.express as px
from src.fetch_data import load_data_from_lag_to_today
from src.process_data import col_date, col_donnees, main_process, fic_export_data
import logging
import os
import glob
 
logging.basicConfig(level=logging.INFO)
 
 
LAG_N_DAYS: int = 7
 
# * INIT REPO FOR DATA
os.makedirs("data/raw/", exist_ok=True)
os.makedirs("data/interim/", exist_ok=True)
 
# * remove outdated json files
for file_path in glob.glob("data/raw/*json"):
    try:
        os.remove(file_path)
    except FileNotFoundError as e:
        logging.warning(e)
 
# plt.switch_backend("TkAgg")
 
# Title for your app
st.title("Data Visualization App")
 
 
# Load data from CSV
@st.cache_data(
    ttl=15 * 60
)  # This decorator caches the data to prevent reloading on every interaction.
def load_data(lag_days: int):
    load_data_from_lag_to_today(lag_days)
    main_process()
    data = pd.read_csv(
        fic_export_data, parse_dates=[col_date]
    )  # Adjust 'DateColumn' to your date column name*
    return data
 
 
# Assuming your CSV is named 'data.csv' and is in the same directory as your app.py
df = load_data(LAG_N_DAYS)
 
# Creating a line chart
st.subheader("Line Chart of Numerical Data Over Time")
 
# Select the numerical column to plot
# This lets the user select a column if there are multiple numerical columns available
# numerical_column = st.selectbox('Select the column to visualize:', df.select_dtypes(include=['float', 'int']).columns)
numerical_column = col_donnees
 
# ! Matplotlib - Create the plot
# fig, ax = plt.subplots()
# ax.plot(df[col_date], df[numerical_column])  # Adjust 'DateColumn' to your date column name
# ax.set_xlabel('Time')
# ax.set_ylabel(numerical_column)
# ax.set_title(f'Time Series Plot of {numerical_column}')
# # Display the plot
# st.pyplot(fig)
 
# ! Plotly
# Create interactive line chart using Plotly
fig = px.line(df, x=col_date, y=col_donnees, title="Consommation en fonction du temps")
st.plotly_chart(fig)
 
# Ajouter une colonne d'heure
df['Hour'] = df[col_date].dt.hour
 
# Grouper les données par heure et calculer la moyenne de la consommation
hourly_avg_consumption = df.groupby('Hour')[col_donnees].mean().reset_index()
 
# Créer un diagramme de la moyenne de la consommation par heure
st.subheader("Moyenne de la consommation par heure de la journée")
fig_hourly_avg = px.bar(hourly_avg_consumption, x='Hour', y=col_donnees, title="Moyenne de la consommation par heure de la journée")
st.plotly_chart(fig_hourly_avg)

# Calculer la somme totale de la consommation
    somme_consommation = sum(consommation_par_heure)
    
# Calculer le nombre total d'heures
    nombre_heures = len(consommation_par_heure)
    
# Calculer la consommation moyenne
    consommation_moyenne = somme_consommation / nombre_heures
    
    return consommation_moyenne

# Exemple de données de consommation pour chaque heure de la journée (en kWh)
consommation_par_heure = [2.5, 3.2, 2.8, 2.9, 3.5, 4.1, 5.2, 6.3, 7.2, 8.5, 9.1, 8.7]

# Calculer la consommation moyenne
consommation_moyenne_journee = calculer_consommation_moyenne(consommation_par_heure)

print("La consommation moyenne en une journée est de:", consommation_moyenne_journee, "kWh")
