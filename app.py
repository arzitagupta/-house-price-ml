import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("house_model.pkl", "rb"))

st.title("House Price Predictor")

area = st.number_input("Area (sqft)")
bedrooms = st.number_input("Bedrooms")
bathrooms = st.number_input("Bathrooms")
age = st.number_input("House Age")

if st.button("Predict Price"):
    features = np.array([[area, bedrooms, bathrooms, age]])
    price = model.predict(features)

    st.success(f"Estimated Price: ₹ {int(price[0])}")
