import joblib
import streamlit as st
import numpy as np

model = joblib.load('model_opossum.joblib')['model']
scaler = joblib.load('model_opossum.joblib')['scaler']

def age_determine(model, hdlngth, skullw, totlngth, eye, chest, belly):
    
    age = {0:'0', 1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9', 10:'10'}
    x = np.array([hdlngth, skullw, totlngth, eye, chest, belly]).reshape(1,6)
    x_scaled = scaler.transform(x)
    return age[model.predict(x_scaled)[0]]

st.title("Opossum")
hdlngth = st.slider('Taille de la tête (en mm)', 70.0, 105.0, format="%f")
skullw = st.slider('Largeur du crâne (en cm)', 50.0, 80.0, format="%f")
totlngth = st.slider('Longueur totale (en cm)', 65.0, 105.0, format="%f")
eye = st.slider('Taille des yeux (en mm)', 10.0, 20.0, format="%f")
chest = st.slider('Tour de poitrine (en cm)', 15.0, 40.0, format="%f")
belly = st.slider('Tour de ventre (en cm)', 15.0, 50.0, format="%f")

prediction = age_determine(model, hdlngth, skullw, totlngth, eye, chest, belly)

st.write("L'opossum avec ces caractéristiques a :", prediction,"ans")