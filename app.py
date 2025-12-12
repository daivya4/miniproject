import streamlit as st
import flood_prediction
import cloudburst_prediction

st.set_page_config(page_title="Disaster Prediction System", page_icon="⚠️", layout="wide")

st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to:", ["Flood Prediction", "Cloudburst Prediction"])

if page == "Flood Prediction":
    flood_prediction.render_flood_page()

elif page == "Cloudburst Prediction":
    cloudburst_prediction.render_cloudburst_page()
