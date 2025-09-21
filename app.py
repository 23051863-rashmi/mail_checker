# app.py
import streamlit as st
import pickle

# Load the trained model and vectorizer saved from Colab
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Title of the app
st.title("📧 Spam Email Detector")

# Input area for user to enter email text
user_input = st.text_area("Enter the email content you want to check:")

# Button to trigger prediction
if st.button("Check Email"):
    # Convert input text to features
    input_features = vectorizer.transform([user_input])
    
    # Make prediction
    prediction = model.predict(input_features)
    
    # Display result
    if prediction[0] == 1:
        st.error("⚠️ This email is Spam!")
    else:
        st.success("✅ This email is Not Spam")
