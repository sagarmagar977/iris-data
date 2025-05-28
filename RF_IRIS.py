import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from PIL import Image  # <-- This is the missing import

# Load data and model
iris = load_iris()
rf = RandomForestClassifier()
rf.fit(iris.data, iris.target)

# Input sliders
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.0)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 5.0, 3.0)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.0)

input_data = [[sepal_length, sepal_width, petal_length, petal_width]]

if st.button('Predict'):
    prediction = rf.predict(input_data)
    species = iris.target_names[prediction][0].lower()
    
    # Display in VERY LARGE text
    st.markdown(f"""
    <h1 style='text-align: center; font-size: 48px; color: #FF6347;'>
        Predicted Species: {species.capitalize()}
    </h1>
    """, unsafe_allow_html=True)
    
    # Display corresponding flower image
    try:
        image = Image.open(f"{species.capitalize()}.png")
        st.image(image, 
                caption=f"Iris {species.capitalize()}",
                width=400
               )
    except FileNotFoundError:
        st.error(f"Image file '{species}.png' not found!")
        st.write("Required image files:")
        st.write("- setosa.png")
        st.write("- versicolor.png") 
        st.write("- virginica.png")