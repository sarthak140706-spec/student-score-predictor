import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

st.title("🎓 Student Score Predictor (TensorFlow Model)")
st.write("Enter study hours and get the predicted score!")

# Load the pre-trained model
model = load_model("student_score.h5")

hours = st.number_input("Enter number of study hours:", min_value=0.0, step=0.5)

if st.button("Predict Score"):
    pred = model.predict(np.array([[hours]], dtype=float))[0][0]
    st.success(f"📘 Predicted Score: {pred:.2f}")
