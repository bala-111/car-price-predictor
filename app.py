import streamlit as st
import pandas as pd
import numpy as np
import importlib.util
import subprocess
import sys
import os

# -------------------------------------------------------------------
# 🧩 Ensure joblib is available (auto-install if missing)
# -------------------------------------------------------------------
if importlib.util.find_spec("joblib") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "joblib", "-q"])
import joblib

# -------------------------------------------------------------------
# ⚙️ Streamlit Page Configuration
# -------------------------------------------------------------------
st.set_page_config(page_title="Used Car Price Predictor", page_icon="🚗", layout="centered")
st.title("🚗 Used Car Price Prediction App")
st.caption("Predict resale price using ML model trained on CarDekho dataset")

# -------------------------------------------------------------------
# 🧠 Load Trained Model
# -------------------------------------------------------------------
@st.cache_resource
def load_model():
    model_path = "used_car_price_model.pkl"
    if not os.path.exists(model_path):
        st.error("❌ Model file not found! Please upload 'used_car_price_model.pkl'.")
        st.stop()
    model = joblib.load(model_path)
    return model

model = load_model()
st.success("✅ Model loaded successfully!")

# -------------------------------------------------------------------
# 📘 Brand–Model–Specs Dictionary (Full List)
# -------------------------------------------------------------------
@st.cache_data
def get_car_specs():
    car_specs = {
        "Maruti": {
            "Alto": [800, 48, 5, 5],
            "Swift": [1200, 83, 5, 5],
            "Baleno": [1200, 90, 5, 5],
            "Ciaz": [1500, 103, 5, 5],
            "Wagon R": [1000, 68, 5, 5],
            "Dzire": [1200, 89, 5, 5],
        },
        "Hyundai": {
            "i10": [1200, 80, 5, 5],
            "i20": [1200, 88, 5, 5],
            "Creta": [1500, 113, 5, 5],
            "Verna": [1500, 115, 5, 5],
            "Venue": [1200, 118, 5, 5],
            "Grand": [1200, 83, 5, 5],
        },
        "Honda": {
            "City": [1500, 119, 5, 5],
            "Amaze": [1200, 87, 5, 5],
            "Jazz": [1200, 89, 5, 5],
        },
        "Tata": {
            "Tiago": [1200, 84, 5, 4],
            "Nexon": [1500, 108, 5, 4],
            "Altroz": [1200, 88, 5, 4],
            "Harrier": [2000, 170, 5, 4],
        },
        "Mahindra": {
            "Bolero": [1500, 75, 7, 4],
            "Scorpio": [2200, 140, 7, 4],
            "Thar": [2200, 130, 4, 4],
            "XUV500": [2200, 155, 7, 4],
            "XUV700": [1999, 197, 7, 4],
        },
        "Toyota": {
            "Innova": [2400, 150, 7, 5],
            "Fortuner": [2800, 175, 7, 5],
            "Glanza": [1200, 89, 5, 5],
        },
        "Ford": {
            "Ecosport": [1500, 121, 5, 3],
            "Figo": [1200, 95, 5, 3],
            "Aspire": [1200, 86, 5, 3],
        },
        "Volkswagen": {
            "Vento": [1600, 105, 5, 4],
            "Polo": [1200, 75, 5, 4],
        },
        "BMW": {
            "X1": [2000, 190, 5, 2],
            "3 Series": [2000, 255, 5, 2],
            "5 Series": [3000, 265, 5, 2],
        },
        "Mercedes-Benz": {
            "C-Class": [2000, 200, 5, 2],
            "E-Class": [2000, 250, 5, 2],
            "GLA": [2000, 190, 5, 2],
        },
        "Audi": {
            "A3": [1400, 150, 5, 2],
            "A4": [2000, 190, 5, 2],
            "Q3": [2000, 180, 5, 2],
        },
        "Kia": {
            "Seltos": [1500, 115, 5, 4],
            "Sonet": [1200, 83, 5, 4],
        },
        "MG": {
            "Hector": [1500, 141, 5, 3],
            "ZS EV": [0, 176, 5, 3],
        },
        "Jeep": {
            "Compass": [2000, 170, 5, 3],
        },
        "Renault": {
            "Duster": [1500, 106, 5, 3],
            "Kwid": [1000, 68, 5, 3],
        },
        "Nissan": {
            "Magnite": [1000, 98, 5, 3],
        },
        "Volvo": {
            "XC40": [2000, 190, 5, 2],
            "XC60": [2000, 250, 5, 2],
        },
        "Jaguar": {
            "XF": [2000, 250, 5, 2],
            "XE": [2000, 247, 5, 2],
        },
        "Mini": {
            "Cooper": [1500, 134, 4, 1],
        },
        "Land Rover": {
            "Discovery": [2000, 240, 7, 2],
        },
        "Ferrari": {
            "488": [3900, 660, 2, 1],
        },
        "Rolls-Royce": {
            "Ghost": [6600, 563, 5, 1],
        },
    }
    return car_specs

car_specs = get_car_specs()
st.success(f"✅ Loaded {len(car_specs)} brands and {sum(len(v) for v in car_specs.values())} models")

# -------------------------------------------------------------------
# 🏷️ Input Section
# -------------------------------------------------------------------
brand = st.selectbox("Select Car Brand", sorted(car_specs.keys()))

if brand:
    models = sorted(car_specs[brand].keys())
    car_model = st.selectbox(f"Select {brand} Model", models)
    if car_model:
        engine, power, seats, brand_score = car_specs[brand][car_model]
        st.info(f"**Auto-filled Specs:** {engine} CC | {power} BHP | {seats} Seats | Brand Score {brand_score}")

col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Vehicle Age (yrs)", 0, 25, 5)
with col2:
    km = st.number_input("Km Driven", 0, 300000, 40000, step=500)
with col3:
    mileage = st.number_input("Mileage (km/l)", 5.0, 40.0, 18.0, step=0.1)

col4, col5 = st.columns(2)
with col4:
    fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
with col5:
    trans = st.selectbox("Transmission", ["Manual", "Automatic"])

diesel_flag   = int(fuel.lower() == "diesel")
electric_flag = int(fuel.lower() == "electric")
cng_lpg_flag  = int(fuel.lower() in ["cng", "lpg"])
auto_flag     = int(trans.lower() == "automatic")

# -------------------------------------------------------------------
# 🔮 Prediction
# -------------------------------------------------------------------
if st.button("Predict Price 💰"):
    if not brand or not car_model:
        st.error("⚠️ Please select both Brand and Model.")
    else:
        features = np.array([[age, km, mileage, engine, power, seats,
                              brand_score, diesel_flag, electric_flag,
                              cng_lpg_flag, auto_flag]])
        pred = model.predict(features)[0]
        st.subheader(f"🎯 Estimated Resale Price: ₹ {pred:,.0f}")
        st.caption("Model: Random Forest (Optimized)")

st.markdown("---")
st.markdown("<p style='text-align:center;'>Made with ❤️ using Streamlit on Hugging Face</p>", unsafe_allow_html=True)
