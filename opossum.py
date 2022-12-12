import joblib
import streamlit as st
import numpy as np

model = joblib.load('model_opossum.joblib')['model']
scaler = joblib.load('model_opossum.joblib')['scaler']

def age_determine(model, hdlngth, skullw, totlngth, eye, chest, belly):
    
    x = np.array([hdlngth, skullw, totlngth, eye, chest, belly]).reshape(1,6)
    x_scaled = scaler.transform(x)
    return model.predict(x_scaled)[0]

st.title("Âge d'un opossum")
head = st.slider('Taille de la tête (en mm)', 70.0, 105.0, format="%f")
skull = st.slider('Largeur du crâne (en cm)', 50.0, 80.0, format="%f")
totalL = st.slider('Longueur totale (en cm)', 65.0, 105.0, format="%f")
eye_dist = st.slider('Taille des yeux (en mm)', 10.0, 20.0, format="%f")
chest_size = st.slider('Tour de poitrine (en cm)', 15.0, 40.0, format="%f")
belly_size = st.slider('Tour de ventre (en cm)', 15.0, 50.0, format="%f")

prediction = age_determine(model, head, skull, totalL, eye_dist, chest_size, belly_size)

st.write("L'opossum avec ces caractéristiques a :", round(prediction, 1), "ans")