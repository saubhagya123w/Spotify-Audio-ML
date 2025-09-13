# Spotify Audio ML — Genre Classification & Popularity Analysis

## Project Overview
This project focuses on analyzing and classifying Spotify tracks using their audio features. It demonstrates end-to-end machine learning workflow, including data cleaning, feature analysis, model training, evaluation, and business insights. The primary goal is to **classify tracks by genre** and **understand key factors affecting track popularity**.

---

## Problem Statement
Spotify has millions of tracks, and classifying them into genres using audio features allows for better music recommendations, playlist curation, and market insights. The project aims to build a **machine learning model** that predicts a track’s genre from its audio characteristics.

---

## Business Use Cases
1. **Recommendation System** — Suggest tracks to users based on similar audio features.  
2. **Genre Classification** — Automatically label new tracks by genre for organization and discovery.  
3. **Music Analytics** — Identify trends in audio features and popularity over time.  

---

## Dataset
- Source: Spotify Web API  
- Cleaned and processed for modeling  
- Key columns:
  - `track_id`, `track_name`, `artists`, `album_name`
  - `popularity` (0–100)
  - Audio features: `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `duration_sec`
  - `track_genre` (target for classification)

---

## Approach
1. **Data Cleaning & EDA**
   - Normalized column names, converted types
   - Handled missing values and duplicates
   - Visualized distributions and correlations of features
2. **Feature Engineering**
   - Duration converted to seconds
   - Scaled numeric features for model training
3. **Model Training**
   - RandomForestClassifier for genre prediction
   - Memory-efficient parameters: `n_estimators=50`, `max_depth=15`
4. **Model Evaluation**
   - Accuracy, Precision, Recall, F1-score
   - Confusion matrix to analyze genre misclassifications
5. **Feature Importance Analysis**
   - Identified top features influencing genre classification
   - Visualized with bar plots
6. **Optional PCA/Clustering**
   - Visualized track clusters in 2D feature space

---

## Key Results
- **Genre Classification:** High accuracy in predicting track genres using audio features.  
- **Feature Importance:** Danceability, energy, loudness, and tempo are the most influential features.  
- **Insights:** Tracks with higher energy and danceability tend to belong to Pop and Dance genres, while high instrumentalness and acousticness correlate with Jazz and Classical tracks.  

---

## Technology Used
- **Programming Language:** Python  
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn  
- **Data Handling:** CSV, parquet  
- **Machine Learning:** RandomForestClassifier, LabelEncoder, StandardScaler  

---

## Project Structure
Spotify-Project/
├── data/
│ └── spotify_cleaned.csv # Cleaned dataset
├── models/
│ ├── rf_genre_model.joblib # Trained RandomForest model
│ └── genre_label_encoder.joblib # LabelEncoder for genres
├── notebooks/
│ ├── 01_data_cleaning.ipynb # EDA & data cleaning
│ └── 02_genre_classification.ipynb # Modeling, evaluation, feature importance
├── README.md
└── requirements.txt

yaml
Copy code

---

## References
- Spotify Web API: https://developer.spotify.com/documentation/web-api/  
- scikit-learn documentation: https://scikit-learn.org/stable/  
- Python libraries: pandas, numpy, matplotlib, seaborn  

---

## Contact
- **Name:** Saubhagya Mishra  
- **Email:** saubhagyamishraa@gmail.com  
