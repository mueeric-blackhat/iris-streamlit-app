import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load('iris_model.pkl')

# Set title and description
st.set_page_config(page_title="Iris Species Classifier", layout="centered")
st.title("🌸 Iris Species Classifier")
st.write("Enter flower measurements to predict the Iris species.")

# Input sliders
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Predict button
if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    species = ['Setosa', 'Versicolor', 'Virginica'][prediction[0]]
    st.success(f"🌼 Predicted Species: **{species}**")
