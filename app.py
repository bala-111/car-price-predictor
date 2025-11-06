import streamlit as st
import pandas as pd
import numpy as np
import gzip, joblib

# ---------------------------------------------------------
# ⚙️ App Configuration
# ---------------------------------------------------------
st.set_page_config(page_title="Used Car Price Predictor", page_icon="🚗")
st.title("🚗 Used Car Price Prediction App")
st.caption("Predict resale price using ML model trained on Cardekho dataset")

# ---------------------------------------------------------
# 🧠 Load Trained Model
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    with gzip.open("used_car_price_model.pkl.gz", "rb") as f:
        model = joblib.load(f)
    return model

model = load_model()
st.success("✅ Model loaded successfully!")

# ---------------------------------------------------------
# 📂 Load Dataset (for brand-model mapping)
# ---------------------------------------------------------
@st.cache_data
def load_specs():
    df = pd.read_csv("Cardekho.csv")
    df.columns = df.columns.str.strip().str.lower()

    brand_scores = {
        'toyota':5,'maruti':5,'honda':5,'hyundai':5,
        'tata':4,'mahindra':4,'kia':4,'skoda':4,'volkswagen':4,
        'ford':3,'renault':3,'nissan':3,'jeep':3,'mg':3,'fiat':3,
        'audi':2,'bmw':2,'mercedes-benz':2,'volvo':2,'jaguar':2,'land rover':2,'lexus':2,
        'rolls-royce':1,'bentley':1,'ferrari':1,'porsche':1,
        'maserati':1,'mini':1,'isuzu':1,'force':1,'datsun':1
    }

    df["brand"] = df["brand"].str.strip().str.title()
    df["model"] = df["model"].str.strip().str.title()
    df["brand_score"] = df["brand"].str.lower().map(brand_scores).fillna(1)

    agg_df = (
        df.groupby(["brand", "model"], as_index=False)
          .agg({"engine":"mean","max_power":"mean","seats":"mean","brand_score":"first"})
          .dropna()
    )

    agg_df["engine"] = agg_df["engine"].round(0).astype(int)
    agg_df["max_power"] = agg_df["max_power"].round(0).astype(int)
    agg_df["seats"] = agg_df["seats"].round(0).astype(int)

    car_specs = {}
    for _, r in agg_df.iterrows():
        car_specs.setdefault(r["brand"], {})[r["model"]] = [
            r["engine"], r["max_power"], r["seats"], int(r["brand_score"])
        ]
    return car_specs

car_specs = load_specs()
st.success(f"✅ Loaded {len(car_specs)} brands and {sum(len(v) for v in car_specs.values())} models")

# ---------------------------------------------------------
# 🏷️ Input Section
# ---------------------------------------------------------
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
    fuel = st.selectbox("Fuel Type", ["Petrol","Diesel","CNG","LPG","Electric"])
with col5:
    trans = st.selectbox("Transmission", ["Manual","Automatic"])

diesel_flag   = int(fuel.lower()=="diesel")
electric_flag = int(fuel.lower()=="electric")
cng_lpg_flag  = int(fuel.lower() in ["cng","lpg"])
auto_flag     = int(trans.lower()=="automatic")

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
        st.caption("Model: Random Forest + Stacking Ensemble")

st.markdown("---")
st.markdown("<p style='text-align:center;'>Made with ❤️ using Streamlit</p>", unsafe_allow_html=True)
