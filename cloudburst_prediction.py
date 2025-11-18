import streamlit as st
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def render_cloudburst_page():

    st.title("🌩 Cloudburst Prediction")
    st.markdown("Predict cloudburst likelihood using Logistic Regression or SVM.")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Cloudburst Settings")
        model_choice = st.selectbox("Select Model", ["Logistic Regression", "SVM"])
        show_probs = st.checkbox("Show Predicted Probability", True)

    root = Path(__file__).parent

    # Safe Model Loading
    def load_model(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except:
            return None

    logreg_cloud = load_model(root / "cloudburst_logreg.pkl")
    svm_cloud = load_model(root / "svm_cloudburst.pkl")
    scaler_cloud = load_model(root / "scaler_cloudburst.pkl")

    # Input UI (same layout style as flood)
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        with st.form("cloudburst_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                temp = st.number_input("Temperature (°C)", value=25.0)
                dew = st.number_input("Dew Point (°C)", value=22.0)

            with col2:
                hum = st.number_input("Relative Humidity (%)", value=80.0)
                slp = st.number_input("Sea Level Pressure (hPa)", value=1010.0)

            with col3:
                cloud_cover = st.number_input("Cloud Cover (%)", value=90.0)
                wind = st.number_input("Wind Speed (km/h)", value=5.0)

            submitted = st.form_submit_button("🔍 Predict Cloudburst")

        st.markdown("</div>", unsafe_allow_html=True)

    if submitted:

        if scaler_cloud is None:
            st.error("❌ Cloudburst scaler/model files not found!")
            return

        input_df = pd.DataFrame([{
            "TEMPERATURE": temp,
            "RELATIVE HUMIDITY": hum,
            "DEWPOINT": dew,
            "SEALEVEL PRESSURE": slp,
            "CLOUD COVER": cloud_cover,
            "WIND SPEED ": wind
        }])

        X_scaled = scaler_cloud.transform(input_df)

        model = logreg_cloud if model_choice == "Logistic Regression" else svm_cloud
        pred = model.predict(X_scaled)[0]

        prob = None
        try:
            prob = model.predict_proba(X_scaled)[0][1]
        except:
            pass

        col1, col2 = st.columns([2, 3])

        with col1:
            if pred == 1:
                st.error("⚠️ Cloudburst Likely!")
            else:
                st.success("🌤 No Cloudburst Expected")

            if show_probs and prob is not None:
                st.info(f"Probability: **{prob*100:.2f}%**")

        with col2:
            st.subheader("Input Summary")
            st.table(input_df.T.rename(columns={0: "Value"}))

            st.markdown("---")

            # Quick Feature Visualization
            st.subheader("Feature Visualization")
            features = ["TEMPERATURE", "RELATIVE HUMIDITY", "DEWPOINT", "CLOUD COVER"]
            values = [temp, hum, dew, cloud_cover]
            vals_norm = (np.array(values) - np.min(values)) / (np.max(values) - np.min(values) + 1e-6)

            fig, ax = plt.subplots(figsize=(5, 2))
            bars = ax.bar(features, vals_norm)

            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f"{val:.1f}", ha='center', fontsize=8)

            ax.set_ylim(0, 1.15)
            plt.xticks(rotation=15, fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
