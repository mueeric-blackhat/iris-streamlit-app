import streamlit as st
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# 🚀 Train once per server restart
@st.cache_resource
def build_model():
    iris = load_iris()
    X, y = iris.data, iris.target
    clf = DecisionTreeClassifier().fit(X, y)
    return clf, iris.target_names

model, target_names = build_model()

st.set_page_config(page_title="Iris Species Classifier", layout="centered")
st.title("🌸 Iris Species Classifier")
st.write("Enter flower measurements to predict the species.")

# Sliders
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width  = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width  = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

if st.button("Predict"):
    sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    pred   = model.predict(sample)[0]
    st.success(f"🌼 Predicted species: **{target_names[pred].title()}**")
