# -------------------------------
# 1. Define CustomLinearRegressor class
# -------------------------------
class CustomLinearRegressor:
    def __init__(self, learning_rate=0.01, epochs=2000):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X, y):
        pass  # Not needed for Streamlit

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.theta).flatten()

# -------------------------------
# 2. Import Libraries
# -------------------------------
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import plotly.graph_objects as go
import plotly.express as px
import time

# -------------------------------
# 3. Streamlit Page Config
# -------------------------------
st.set_page_config(page_title="DeepPrevent: Predictive Maintenance", layout="wide")
st.title("🛠️ DeepPrevent: Predictive Maintenance Dashboard")

# -------------------------------
# Load Normalization Stats & Dataset
# -------------------------------
stats = pd.read_csv("scaling_stats.csv")
X_mean = stats["mean"].values
X_std = stats["std"].values
df = pd.read_csv("iocl_machinery_dataset.csv")

# Load performance metrics if available
try:
    perf_df = pd.read_csv("model_performance.csv")
except:
    perf_df = None

# -------------------------------
# Sidebar: Model Selection
# -------------------------------
st.sidebar.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSb3G380oUY12-ni7OyEgj6qpbfvalkZ9n2qw&shttps://iocl.com/iocl-logo-types/logos/gif/IndianOilLogo1024x768.gif", width=250)
st.sidebar.header("⚙️ Model Selection")
model_choice = st.sidebar.selectbox(
    "Choose Model for Prediction",
    ["Custom Linear Regression", "Ridge Regression", "Random Forest", "Gradient Boosting"]
)

# Load selected model
if model_choice == "Custom Linear Regression":
    model = joblib.load("trained_model.pkl")
elif model_choice == "Ridge Regression":
    model = joblib.load("ridge_model.pkl")
elif model_choice == "Random Forest":
    model = joblib.load("rf_model.pkl")
else:
    model = joblib.load("gb_model.pkl")

# -------------------------------
# Tabs Layout
# -------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Prediction", "📊 Data Insights", "📈 Model Comparison", "ℹ️ About Project"])

# -------------------------------
# -------------------------------
# 🔍 Tab 1: Prediction + Live Simulation
# -------------------------------
with tab1:
    st.subheader("Enter Machine Sensor Data or Run Live Simulation")
    col1, col2 = st.columns(2)
    with col1:
        temp = st.slider("Temperature (°C)", 50.0, 100.0, 70.0)
        vib = st.slider("Vibration (mm/s)", 0.1, 1.0, 0.45)
    with col2:
        oil = st.slider("Oil Flow (L/min)", 20.0, 40.0, 32.5)
        rpm = st.slider("RPM", 2000, 4000, 2800)

    # Manual Prediction Button
    if st.button("🔍 Predict RUL"):
        with st.spinner("Calculating RUL..."):
            time.sleep(1)
        input_data = np.array([[temp, vib, oil, rpm]])
        input_scaled = (input_data - X_mean) / X_std
        predicted_rul = model.predict(input_scaled)[0]

        # Display Prediction
        st.success(f"🧠 Predicted RUL: **{predicted_rul:.2f} cycles**")
        status = "✅ Good Health" if predicted_rul > 150 else "⚠️ Maintenance Soon" if predicted_rul > 90 else "🚨 Critical!"
        st.write(status)

        # Gauge Chart
        color = "green" if predicted_rul > 150 else "orange" if predicted_rul > 90 else "red"
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=predicted_rul,
            title={'text': "Predicted RUL (cycles)"},
            gauge={'axis': {'range': [0, 250]}, 'bar': {'color': color}}
        ))
        st.plotly_chart(fig_gauge)

    st.markdown("---")

    # ✅ Live Simulation
    st.subheader("📡 Live Monitoring Simulation")
    run_simulation = st.checkbox("Enable Real-Time Simulation")

    if run_simulation:
        st.info("Simulation is running... generating random sensor values every 2 seconds.")
        simulation_chart = st.empty()
        gauge_placeholder = st.empty()

        for i in range(10):  # 10 updates
            # Generate random sensor values
            temp_live = np.random.uniform(55, 95)
            vib_live = np.random.uniform(0.2, 0.8)
            oil_live = np.random.uniform(25, 38)
            rpm_live = np.random.uniform(2200, 3600)

            # Predict RUL
            live_data = np.array([[temp_live, vib_live, oil_live, rpm_live]])
            live_scaled = (live_data - X_mean) / X_std
            live_rul = model.predict(live_scaled)[0]

            # Update Gauge
            color = "green" if live_rul > 150 else "orange" if live_rul > 90 else "red"
            fig_gauge_live = go.Figure(go.Indicator(
                mode="gauge+number",
                value=live_rul,
                title={'text': "Predicted RUL (Live)"},
                gauge={'axis': {'range': [0, 250]}, 'bar': {'color': color}}
            ))
            gauge_placeholder.plotly_chart(fig_gauge_live)

            # Update Trend Line
            sim_df = pd.DataFrame({
                "Step": [i],
                "RUL": [live_rul]
            })
            if i == 0:
                history = sim_df
            else:
                history = pd.concat([history, sim_df], ignore_index=True)

            fig_line = px.line(history, x="Step", y="RUL", title="RUL Trend Over Time", markers=True)
            simulation_chart.plotly_chart(fig_line)

            time.sleep(2)


# -------------------------------
# Tab 2: Data Insights
# -------------------------------
with tab2:
    st.subheader("📊 Feature Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("📈 Feature Distribution")
    st.bar_chart(df[['temperature', 'vibration', 'oil_flow', 'rpm']].mean())

# -------------------------------
# Tab 3: Model Comparison
# -------------------------------
with tab3:
    if perf_df is not None:
        st.subheader("📈 Model Performance Metrics")
        st.dataframe(perf_df.style.format({
            "R2": "{:.4f}", "MAE": "{:.2f}", "RMSE": "{:.2f}"
        }))
        best_model = perf_df.loc[perf_df["R2"].idxmax(), "Model"]
        st.success(f"🏆 Best Model: **{best_model}**")

        # Bar Chart
        perf_melted = perf_df.melt(id_vars="Model", value_vars=["R2", "MAE", "RMSE"], var_name="Metric", value_name="Score")
        fig = px.bar(perf_melted, x="Model", y="Score", color="Metric", barmode="group",
                     title="Model Performance Metrics", text="Score")
        st.plotly_chart(fig)
    else:
        st.warning("No performance data found. Upload model_performance.csv")

# -------------------------------
# Tab 4: About Project
# -------------------------------
with tab4:
    st.markdown("""
    ### 🛠️ About DeepPrevent
    **DeepPrevent** is an intelligent predictive maintenance solution for IOCL machinery.
    - **Goal:** Predict Remaining Useful Life (RUL) using sensor data.
    - **Models Used:** Custom Linear Regression (from scratch), Ridge, Random Forest, Gradient Boosting.
    - **Features:** Temperature, Vibration, Oil Flow, RPM.

    **Key Highlights:**
    - Live RUL prediction.
    - Interactive dashboards.
    - Model comparison and performance visualization.
    """)
