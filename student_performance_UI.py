import streamlit as st
import joblib
import numpy as np

# Load models
regression_model = joblib.load("student_score_predictor.pkl")
classification_model = joblib.load("student_pass_fail_predictor.pkl")

# Preprocess input
def preprocess_input(gender, study_hours, attendance, past_score,
                     internet, extra, parental_education):
    gender = 1 if gender == 'Male' else 0
    internet = 1 if internet == 'Yes' else 0
    extra = 1 if extra == 'Yes' else 0

    hs = 1 if parental_education == 'High School' else 0
    masters = 1 if parental_education == 'Masters' else 0
    phd = 1 if parental_education == 'PhD' else 0

    features = [gender, float(study_hours), float(attendance), float(past_score),
                internet, extra, hs, masters, phd]
    return np.array(features).reshape(1, -1)

# --- Streamlit UI ---
st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓")
st.title("🎓 Student Performance Predictor")
st.markdown("Predict a student's **final score** and **pass/fail status** based on various factors.")

# Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
study_hours = st.number_input("Study Hours per Week", min_value=0.0, step=0.5)
attendance = st.slider("Attendance Rate (%)", min_value=0, max_value=100, value=75)
past_score = st.number_input("Past Exam Score", min_value=0.0, max_value=100.0, step=0.5)

internet = st.radio("Internet at Home", ["Yes", "No"])
extra = st.radio("Extracurricular Activities", ["Yes", "No"])
parental_education = st.selectbox("Parental Education Level", ["High School", "Bachelor's", "Masters", "PhD"])

# Predict Button
if st.button("🔮 Predict"):
    try:
        processed = preprocess_input(gender, study_hours, attendance, past_score,
                                     internet, extra, parental_education)

        final_score = regression_model.predict(processed)[0]
        pass_fail = classification_model.predict(processed)[0]
        status = "Pass" if pass_fail == 1 else "Fail"

        st.success(f"📊 Final Score: `{final_score:.2f}`")
        st.markdown(f"📌 **Status:** `{status}`", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"⚠️ An error occurred: {e}")
