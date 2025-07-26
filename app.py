import streamlit as st
import joblib
import numpy as np
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt

# App settings
st.set_page_config(page_title="Introvert or not?", page_icon="🧠", layout="centered")

# Load the trained model
model = joblib.load("personality_model.pkl")

# --- Title Section ---
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>Introvert or not?</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Is this person an <b>Introvert</b> or an <b>Extrovert</b>? Let’s find out!</p>", unsafe_allow_html=True)
st.markdown("---")

# --- Input Form ---
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        time_alone = st.slider("🕒 Time Spent Alone", 0, 10, 5)
        stage_fear = st.radio("🎤 Stage Fear", ["No", "Yes"])
        going_outside = st.slider("🚶 Going Outside", 0, 10, 5)
        post_freq = st.slider("📱 Post Frequency", 0, 10, 5)

    with col2:
        social_attendance = st.slider("🎉 Social Event Attendance", 0, 10, 5)
        drained_social = st.radio("😵 Drained After Socializing", ["No", "Yes"])
        friends_circle = st.slider("👥 Friends Circle Size", 0, 20, 5)

    submit_btn = st.form_submit_button("🔍 Let' Go!")

# Encode inputs
stage_fear_val = 1 if stage_fear == "Yes" else 0
drained_val = 1 if drained_social == "Yes" else 0

input_data = np.array([[time_alone, stage_fear_val, social_attendance,
                        going_outside, drained_val, friends_circle, post_freq]])

# --- Output Result ---
if submit_btn:
    # Get prediction and probabilities
    proba = model.predict_proba(input_data)[0]
    prediction = model.predict(input_data)[0]

    # Confidence for predicted class
    confidence = proba[prediction] * 100

    # Text Output
    if prediction == 1:
        st.markdown(f"<h2 style='color: #FF4B4B;'>🧘 You are an <u>Introvert</u></h2>", unsafe_allow_html=True)
        st.info("You enjoy solitude, reflection, and peace ✨")
    else:
        st.markdown(f"<h2 style='color: #00C851;'>🎉 You are an <u>Extrovert</u></h2>", unsafe_allow_html=True)
        st.success("You thrive in company, energy, and social vibes 🔥")

    # Show confidence score
    st.markdown(f"<p style='font-size: 18px;'>🔎 Confidence: <b>{confidence:.2f}%</b></p>", unsafe_allow_html=True)

    # Prepare DataFrame for bar chart
    df_proba = pd.DataFrame({
        'Personality': ['Extrovert', 'Introvert'],
        'Probability (%)': [proba[0] * 100, proba[1] * 100]
    })

    # Pie Chart
    labels = ['Extrovert', 'Introvert']
    colors = ['#00C851', '#FF4444']
    explode = [0.05, 0.05]  # for slight separation

    fig, ax = plt.subplots()
    ax.pie([proba[0], proba[1]], labels=labels, autopct='%1.1f%%', startangle=90,
        colors=colors, explode=explode, shadow=True)
    ax.axis('equal')  # Equal aspect ratio

    st.markdown("### 🧁 Personality Breakdown")
    st.pyplot(fig)

    st.balloons()

