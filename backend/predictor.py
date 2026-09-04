import os
import joblib
import numpy as np

from utils.preprocess import preprocess_input

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "heart_disease_model.pkl"
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at: {MODEL_PATH}"
    )

loaded_model = joblib.load(MODEL_PATH)

model = loaded_model["model"]
scaler = loaded_model["scaler"]
feature_names = loaded_model["feature_names"]


def predict_heart_disease(patient_data):
    """
    Predict heart disease using the trained model.
    """

    processed_data = preprocess_input(patient_data, scaler)

    prediction = model.predict(processed_data)[0]

    probability = model.predict_proba(processed_data)[0][1]

    return {
        "prediction": int(prediction),
        "probability": round(float(probability), 4)
    }