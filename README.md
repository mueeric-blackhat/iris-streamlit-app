# 🌸 Iris Species Classifier

A simple machine learning web app built using **Scikit-learn** and **Streamlit** to predict the species of an iris flower based on sepal and petal measurements.

## 📌 Project Overview

This project applies classical machine learning (Decision Tree Classifier) to the famous **Iris dataset** to classify iris flowers into one of three species:

- Setosa
- Versicolor
- Virginica

The model was trained in a Jupyter Notebook and deployed using Streamlit Cloud.

---

## 📂 Files in This Repository

| File                  | Description                                      |
|-----------------------|--------------------------------------------------|
| `iris_classifier.ipynb` | Jupyter Notebook: full training workflow        |
| `iris_model.pkl`      | Saved Decision Tree model                        |
| `app.py`              | Streamlit app script                             |
| `requirements.txt`    | Python dependencies for the app                  |
| `README.md`           | Project description and setup instructions       |

---

## 🚀 Try the Live App

👉 **Live Demo:**
(https://iris-app-appgit-nmyn4cuxctkymrncodcrs2.streamlit.app/)
---

## 🛠️ How to Run Locally

To test the app on your own machine:

```bash
# 1. Clone the repository
git clone https://github.com/your-username/iris-streamlit-app.git
cd iris-streamlit-app

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
✅ Features
📊 Built with Scikit-learn and trained on the Iris dataset

🔍 Predicts iris species based on user input

🌐 Deployed using Streamlit Cloud

📈 Evaluated using accuracy, precision, and recall

🎨 Clean and simple user interface

📊 ML Model Info
Algorithm: Decision Tree Classifier

Framework: Scikit-learn

Dataset: load_iris() from Scikit-learn

Metrics Used: Accuracy, Precision, Recall, Classification Report

📚 Requirements txt
streamlit
scikit-learn
numpy

Install with:

bash

pip install -r requirements.txt
