import streamlit as st
import numpy as np
import pickle
import pandas as pd

csv_file_path = r'C:\Users\mahap\OneDrive\Desktop\C++,JS python codes\.vscode\ML-DL-NLP\myenv\CompleteSVRproject\Salary Data.csv'
data = pd.read_csv(csv_file_path)

st.title("Salary Prediction App")
st.write("### Data Overview")
st.write(data.head())

independent_features = data.columns[:-1]

# Collect user input
st.write("### Enter Details")
user_input = {}

for feature in independent_features:
    if feature == 'Education Level':
        continue  # Skip for now
    if data[feature].dtype == 'object':
        user_input[feature] = st.selectbox(f"{feature}", options=data[feature].unique())
    else:
        user_input[feature] = st.number_input(f"{feature}", value=0.0)

st.write("### Education Level")
education_level = st.selectbox("Select your Education Level:", options=['Bachelor\'s', 'Master\'s', 'PhD'])

# Create keys for "Education Level_Master's" and "Education Level_PhD"
if education_level == 'Bachelor\'s':
    user_input['Education Level_Master\'s'] = 0
    user_input['Education Level_PhD'] = 0
elif education_level == 'Master\'s':
    user_input['Education Level_Master\'s'] = 1
    user_input['Education Level_PhD'] = 0
elif education_level == 'PhD':
    user_input['Education Level_Master\'s'] = 0
    user_input['Education Level_PhD'] = 1

if st.button("Predict"):
    try:
        # Load scaler, model, and encoder from pickle files
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        with open('Salary_Prediction.pkl', 'rb') as file:
            model = pickle.load(file)
        with open('Job_encoder.pkl', 'rb') as file:
            encoder = pickle.load(file)

        # Convert user input to a DataFrame
        input_df = pd.DataFrame([user_input])
        input_df=encoder.transform(input_df)
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)
        st.write(f"### Prediction: {prediction[0]}")

    except Exception as e:
        st.error(f"Error occurred: {str(e)}")
