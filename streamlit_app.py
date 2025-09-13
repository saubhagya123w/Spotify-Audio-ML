import streamlit as st
import numpy as np
import joblib

# Load memory-efficient RandomForest model, scaler, and label encoder
model = joblib.load("models/rf_genre_model.joblib")
scaler = joblib.load("models/scaler.joblib")
le = joblib.load("models/genre_label_encoder.joblib")

st.title("Spotify Genre Classification")
st.write("Predict the genre of a track based on its audio features.")

# User input sliders
danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
energy = st.slider("Energy", 0.0, 1.0, 0.5)
loudness = st.slider("Loudness (dB)", -60.0, 0.0, -10.0)
speechiness = st.slider("Speechiness", 0.0, 1.0, 0.05)
acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5)
instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0)
liveness = st.slider("Liveness", 0.0, 1.0, 0.1)
valence = st.slider("Valence", 0.0, 1.0, 0.5)
tempo = st.slider("Tempo (BPM)", 0.0, 250.0, 120.0)
duration_sec = st.slider("Duration (seconds)", 30.0, 600.0, 180.0)

# Prediction
if st.button("Predict Genre"):
    # Prepare feature array
    features = np.array([[danceability, energy, loudness, speechiness,
                          acousticness, instrumentalness, liveness, valence,
                          tempo, duration_sec]])
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict genre
    pred_label = model.predict(features_scaled)
    pred_genre = le.inverse_transform(pred_label)[0]
    
    st.success(f"Predicted Genre: **{pred_genre}**")
