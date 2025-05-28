import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load iris data and train model once
iris = load_iris()
X, y = iris.data, iris.target
rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)

st.title("Iris Flower Species Prediction")

# User inputs for features
sepal_length = st.slider('Sepal length (cm)', float(X[:,0].min()), float(X[:,0].max()), float(X[:,0].mean()))
sepal_width = st.slider('Sepal width (cm)', float(X[:,1].min()), float(X[:,1].max()), float(X[:,1].mean()))
petal_length = st.slider('Petal length (cm)', float(X[:,2].min()), float(X[:,2].max()), float(X[:,2].mean()))
petal_width = st.slider('Petal width (cm)', float(X[:,3].min()), float(X[:,3].max()), float(X[:,3].mean()))

# Prepare input data for prediction
input_data = [[sepal_length, sepal_width, petal_length, petal_width]]

if st.button('Predict'):
    prediction = rf.predict(input_data)
    species = iris.target_names[prediction][0]
    st.write(f"Predicted Iris species: **{species}**")
