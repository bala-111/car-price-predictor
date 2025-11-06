import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------------------------------------
# ⚙️ App Configuration
# ---------------------------------------------------------
st.set_page_config(page_title="Used Car Price Predictor", page_icon="🚗")
st.title("🚗 Used Car Price Prediction App")
st.caption("Predict resale price using an optimized Random Forest model (120 trees) trained on the CarDekho dataset")

# ---------------------------------------------------------
# 🧠 Load Trained Model (no gzip, smaller size)
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("used_car_price_model.pkl")
    return model

model = load_model()
st.success("✅ ML Model loaded successfully!")

# ---------------------------------------------------------
# 📘 Load Brand–Model Specs CSV (dictionary data)
# ---------------------------------------------------------
@st.cache_data
def load_specs():
    import os

    local_path = "brand_model_specs.csv"
    github_url = "https://raw.githubusercontent.com/bala-111/car-price-predictor/main/brand_model_specs.csv"

    # ✅ Try local file first
    if os.path.exists(local_path):
        df = pd.read_csv(local_path)
        st.info("📂 Loaded local CSV file successfully.")
    else:
        # ✅ Fallback to GitHub raw link
        df = pd.read_csv(github_url)
        st.info("🌐 Loaded CSV from GitHub repository.")

    # --- Clean columns ---
    df.columns = df.columns.str.strip().str.lower()

    # --- Create dictionary ---
    car_specs = {}
    for _, r in df.iterrows():
        brand = r["brand"].strip().title()
        model_name = r["model"].strip().title()
        car_specs.setdefault(brand, {})[model_name] = [
            int(r["engine"]), int(r["max_power"]), int(r["seats"]), int(r["brand_score"])
        ]

    return car_specs


car_specs = load_specs()
st.success(f"✅ Loaded {len(car_specs)} brands and {sum(len(v) for v in car_specs.values())} models")

# ---------------------------------------------------------
# 🏷️ User Inputs
# ---------------------------------------------------------
st.subheader("🧩 Select Car Details")

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

# ---------------------------------------------------------
# 🔮 Prediction
# ---------------------------------------------------------
if st.button("Predict Price 💰"):
    if not brand or not car_model:
        st.error("⚠️ Please select both Brand and Model.")
    else:
        features = np.array([[age, km, mileage, engine, power, seats,
                              brand_score, diesel_flag, electric_flag, cng_lpg_flag, auto_flag]])
        pred = model.predict(features)[0]
        st.subheader(f"🎯 Estimated Resale Price: ₹ {pred:,.0f}")
        st.caption("Model: Random Forest (120 Trees, R² ≈ 0.93)")

st.markdown("---")
st.markdown("<p style='text-align:center;'>Made with ❤️ using Streamlit</p>", unsafe_allow_html=True)
