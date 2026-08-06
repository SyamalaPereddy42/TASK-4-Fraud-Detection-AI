from flask import Flask, request, jsonify
from flask_cors import CORS

from predictor import predict_heart_disease

app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    return jsonify({
        "Project": "Heart Disease Prediction",
        "Status": "Backend Running"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        patient = [
            data["age"],
            data["sex"],
            data["cp"],
            data["trestbps"],
            data["chol"],
            data["fbs"],
            data["restecg"],
            data["thalach"],
            data["exang"],
            data["oldpeak"],
            data["slope"],
            data["ca"],
            data["thal"]
        ]

        result = predict_heart_disease(patient)

        prediction = result["prediction"]
        probability = result["probability"]

        # Risk Level
        if probability < 0.30:
            risk_level = "Low"
        elif probability < 0.70:
            risk_level = "Moderate"
        else:
            risk_level = "High"

        # Result & Recommendation
        if prediction == 1:
            result_text = "Heart Disease Detected"
            recommendation = (
                "Consult a cardiologist. "
                "Maintain a healthy diet, exercise regularly, "
                "avoid smoking, and monitor your blood pressure."
            )
        else:
            result_text = "No Heart Disease"
            recommendation = (
                "No signs of heart disease detected. "
                "Continue a healthy lifestyle and regular check-ups."
            )

        response = {
            "prediction": prediction,
            "probability": probability,
            "result": result_text,
            "risk_level": risk_level,
            "recommendation": recommendation
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )