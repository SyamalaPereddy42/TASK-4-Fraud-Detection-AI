import streamlit as st
import requests

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

API_URL = "http://127.0.0.1:5000/predict"

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>

.main-title{
    text-align:center;
    color:#c1121f;
    font-size:42px;
    font-weight:bold;
}

.sub-title{
    text-align:center;
    color:gray;
    font-size:18px;
}

div[data-testid="stButton"] > button{
    width:100%;
    height:55px;
    background:#c1121f;
    color:white;
    border-radius:10px;
    font-size:20px;
    font-weight:bold;
}

.result-box{
    padding:20px;
    border-radius:10px;
    text-align:center;
    font-size:24px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown("<h1 class='main-title'>❤️ Heart Disease Prediction System</h1>", unsafe_allow_html=True)

st.markdown("<p class='sub-title'>Predict Heart Disease using Machine Learning</p>", unsafe_allow_html=True)

st.divider()

# -----------------------------
# Input Form
# -----------------------------
col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=45
    )

    sex = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    cp = st.selectbox(
        "Chest Pain Type",
        [
            "Typical Angina",
            "Atypical Angina",
            "Non-Anginal Pain",
            "Asymptomatic"
        ]
    )

    trestbps = st.number_input(
        "Resting Blood Pressure (mm Hg)",
        min_value=80,
        max_value=250,
        value=120
    )

    chol = st.number_input(
        "Serum Cholesterol (mg/dl)",
        min_value=100,
        max_value=700,
        value=200
    )

    fbs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        ["No", "Yes"]
    )

    restecg = st.selectbox(
        "Resting ECG",
        [
            "Normal",
            "ST-T Wave Abnormality",
            "Left Ventricular Hypertrophy"
        ]
    )

with col2:

    thalach = st.number_input(
        "Maximum Heart Rate",
        min_value=60,
        max_value=250,
        value=150
    )

    exang = st.selectbox(
        "Exercise Induced Angina",
        ["No", "Yes"]
    )

    oldpeak = st.number_input(
        "ST Depression (Oldpeak)",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1
    )

    slope = st.selectbox(
        "Slope of Peak Exercise ST Segment",
        [
            "Upsloping",
            "Flat",
            "Downsloping"
        ]
    )

    ca = st.selectbox(
        "Number of Major Vessels",
        [0,1,2,3,4]
    )

    thal = st.selectbox(
        "Thalassemia",
        [
            "Normal",
            "Fixed Defect",
            "Reversible Defect",
            "Unknown"
        ]
    )

st.divider()

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("Predict Heart Disease"):

    payload = {

        "age": age,

        "sex": 1 if sex=="Male" else 0,

        "cp": {
            "Typical Angina":0,
            "Atypical Angina":1,
            "Non-Anginal Pain":2,
            "Asymptomatic":3
        }[cp],

        "trestbps": trestbps,

        "chol": chol,

        "fbs": 1 if fbs=="Yes" else 0,

        "restecg":{
            "Normal":0,
            "ST-T Wave Abnormality":1,
            "Left Ventricular Hypertrophy":2
        }[restecg],

        "thalach": thalach,

        "exang": 1 if exang=="Yes" else 0,

        "oldpeak": oldpeak,

        "slope":{
            "Upsloping":0,
            "Flat":1,
            "Downsloping":2
        }[slope],

        "ca": ca,

        "thal":{
            "Normal":0,
            "Fixed Defect":1,
            "Reversible Defect":2,
            "Unknown":3
        }[thal]

    }

    try:

        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:

            result = response.json()

            prediction = result["prediction"]

            probability = result["probability"] * 100

            st.subheader("Prediction Result")

            if prediction == 1:

                st.error("⚠ High Risk of Heart Disease")

            else:

                st.success("✅ Low Risk of Heart Disease")

            st.progress(min(int(probability), 100))

            st.metric(
                "Prediction Confidence",
                f"{probability:.2f}%"
            )

            st.json(result)

        else:

            st.error("Prediction Failed")

            st.write(response.text)

    except Exception as e:

        st.error("Cannot connect to backend.")

        st.write(e)