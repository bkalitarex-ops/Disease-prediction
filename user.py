#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 01:21:15 2026

@author: bhargabkalita
"""

import streamlit as st
import numpy as np
import joblib

# LOAD SAVED FILES
# =========================
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
disease_encoder = joblib.load("disease_encoder.pkl")

# PAGE CONFIGURATION
# =========================
st.set_page_config(
    page_title="Infectious Disease Prediction",
    page_icon="🩺",
    layout="centered"
)

# =========================
# TITLE
# =========================
st.title("🩺 Infectious Disease Prediction System")
st.success(
    "👈 Please fill in patient information from the left sidebar to begin prediction."
)

st.info(
    "Enter symptom severity levels :\n"
    "0 = No Symptom, "
    "1 = Mild, "
    "2 = Moderate, "
    "3 = Severe"
)
st.sidebar.write("Enter patient details to predict possible infectious disease :")

# USER INPUTS
# =========================

age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=25)

# Gender Encoding
# Male = 1, Female = 0

gender_option = st.sidebar.selectbox("Gender", ["Female", "Male"])
gender = 1 if gender_option == "Male" else 0

fever = st.sidebar.selectbox("Fever", [0, 1,2,3])
cough = st.sidebar.selectbox("Cough", [0, 1,2,3])
fatigue = st.sidebar.selectbox("Fatigue", [0, 1,2,3])
headache = st.sidebar.selectbox("Headache", [0, 1,2,3])
muscle_pain = st.sidebar.selectbox("Muscle Pain", [0, 1,2,3])
nausea = st.sidebar.selectbox("Nausea", [0, 1,2,3])
vomiting = st.sidebar.selectbox("Vomiting", [0, 1,2,3])
diarrhea = st.sidebar.selectbox("Diarrhea", [0, 1,2,3])
skin_rash = st.sidebar.selectbox("Skin Rash", [0, 1,2,3])
loss_smell = st.sidebar.selectbox("Loss of Smell",[0,1,2,3])
loss_taste = st.sidebar.selectbox("Loss of Taste",[0,1,2,3])

# PREDICTION BUTTON
# =========================
if st.button("Predict Disease"):

    # Input Data
    input_data = np.array([[
        age,
        gender,
        fever,
        cough,
        fatigue,
        headache,
        muscle_pain,
        nausea,
        vomiting,
        diarrhea,
        skin_rash,
        loss_smell,
        loss_taste
        
    ]])

# Scaling
input_scaled = scaler.transform(input_data)

# Prediction
prediction = model.predict(input_scaled)

# Prediction Probabilities
probabilities = model.predict_proba(input_scaled)[0]

# Decode Disease Name
disease_name = disease_encoder.inverse_transform(prediction)[0]

# Display Main Prediction
st.success(f"MOST LIKELY DISEASE : {disease_name}")


# SHOW PROBABILITIES


st.subheader("Disease Prediction Probabilities")

# Get Disease Names
disease_classes = disease_encoder.classes_

# Create probability dictionary
prob_dict = {}

for disease, prob in zip(disease_classes, probabilities):
    prob_dict[disease] = prob * 100

# Sort probabilities
sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)

# Display probabilities
for disease, prob in sorted_probs:
    st.write(f"{disease}: {prob:.2f}%")
    st.progress(int(prob))