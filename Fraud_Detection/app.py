
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Fraud Detection AI",
    page_icon="💳",
    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: 700;
}

.subtitle {
    font-size: 20px;
    color: #666;
}

.result-box {
    padding: 25px;
    border-radius: 15px;
    margin-top: 15px;
}

.metric-card {
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #ddd;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    model = joblib.load("models/fraud_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    return model, scaler, feature_names


model, scaler, feature_names = load_model()

# ============================================================
# LOAD DATASET
# ============================================================
@st.cache_data
def load_dataset():
    return pd.read_csv("data/creditcard.csv")


df = load_dataset()

# ============================================================
# HEADER
# ============================================================
st.markdown(
    '<div class="main-title">💳 Fraud Detection AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Machine Learning Powered Credit Card Fraud Detection System</div>',
    unsafe_allow_html=True
)

st.divider()

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Choose a section",
    [
        "🏠 Dashboard",
        "🔍 Transaction Prediction",
        "📊 Model Performance",
        "🧪 Test Real Transaction"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    """
Model: Logistic Regression

Dataset:
284,807 transactions

Fraud:
492 transactions

Legitimate:
284,315 transactions
"""
)

# ============================================================
# DASHBOARD
# ============================================================
if page == "🏠 Dashboard":

    st.header("🏠 Fraud Detection Dashboard")

    st.write(
        "This system uses Machine Learning to identify potentially "
        "fraudulent credit card transactions."
    )

    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Transactions",
        f"{len(df):,}"
    )

    col2.metric(
        "Fraud Transactions",
        f"{int(df['Class'].sum()):,}"
    )

    col3.metric(
        "Legitimate Transactions",
        f"{int((df['Class'] == 0).sum()):,}"
    )

    col4.metric(
        "Fraud Rate",
        f"{df['Class'].mean() * 100:.2f}%"
    )

    st.divider()

    # Fraud distribution
    st.subheader("📈 Transaction Distribution")

    distribution = pd.DataFrame({
        "Transaction Type": [
            "Legitimate",
            "Fraudulent"
        ],
        "Count": [
            int((df["Class"] == 0).sum()),
            int((df["Class"] == 1).sum())
        ]
    })

    st.bar_chart(
        distribution.set_index("Transaction Type")
    )

    st.divider()

    st.subheader("🔎 Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

# ============================================================
# TRANSACTION PREDICTION
# ============================================================
elif page == "🔍 Transaction Prediction":

    st.header("🔍 Transaction Prediction")

    st.write(
        "Enter transaction feature values and let the trained "
        "Machine Learning model determine whether the transaction "
        "is potentially fraudulent."
    )

    inputs = {}

    col1, col2, col3 = st.columns(3)

    for i, feature in enumerate(feature_names):

        with [col1, col2, col3][i % 3]:

            inputs[feature] = st.number_input(
                feature,
                value=0.0,
                format="%.6f"
            )

    st.divider()

    if st.button(
        "🚨 Predict Transaction",
        use_container_width=True
    ):

        input_data = pd.DataFrame([inputs])

        input_data = input_data[feature_names]

        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)[0]

        probability = model.predict_proba(
            input_scaled
        )[0][1]

        fraud_percentage = probability * 100

        st.divider()

        st.subheader("📊 Prediction Result")

        if prediction == 1:

            st.error(
                "🚨 FRAUDULENT TRANSACTION DETECTED"
            )

            st.metric(
                "Fraud Probability",
                f"{fraud_percentage:.2f}%"
            )

            st.warning(
                "The model has classified this transaction "
                "as potentially fraudulent."
            )

        else:

            st.success(
                "✅ LEGITIMATE TRANSACTION"
            )

            st.metric(
                "Fraud Probability",
                f"{fraud_percentage:.2f}%"
            )

            st.info(
                "The model predicts that this transaction "
                "is likely legitimate."
            )

# ============================================================
# MODEL PERFORMANCE
# ============================================================
elif page == "📊 Model Performance":

    st.header("📊 Model Performance")

    st.write(
        "Performance metrics obtained from the test dataset."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Accuracy",
        "97.55%"
    )

    col2.metric(
        "Fraud Recall",
        "92%"
    )

    col3.metric(
        "ROC-AUC",
        "97.21%"
    )

    col4.metric(
        "Fraud Detected",
        "90 / 98"
    )

    st.divider()

    st.subheader("🎯 Classification Performance")

    performance = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Fraud Recall",
            "ROC-AUC"
        ],
        "Score": [
            97.55,
            92.00,
            97.21
        ]
    })

    st.bar_chart(
        performance.set_index("Metric")
    )

    st.divider()

    st.subheader("📌 Confusion Matrix")

    confusion = pd.DataFrame(
        [
            [55478, 1386],
            [8, 90]
        ],
        columns=[
            "Predicted Legitimate",
            "Predicted Fraud"
        ],
        index=[
            "Actual Legitimate",
            "Actual Fraud"
        ]
    )

    st.dataframe(
        confusion,
        use_container_width=True
    )

    st.info(
        "Fraud recall is particularly important because it measures "
        "how many actual fraudulent transactions the model successfully detects."
    )

# ============================================================
# TEST REAL TRANSACTION
# ============================================================
elif page == "🧪 Test Real Transaction":

    st.header("🧪 Test a Real Dataset Transaction")

    st.write(
        "Select an actual transaction from the credit card dataset. "
        "The application will automatically load its feature values "
        "and run the trained model."
    )

    # Transaction selection
    transaction_index = st.number_input(
        "Transaction Index",
        min_value=0,
        max_value=len(df) - 1,
        value=0,
        step=1
    )

    transaction = df.iloc[[transaction_index]]

    st.subheader("📋 Selected Transaction")

    st.dataframe(
        transaction,
        use_container_width=True
    )

    actual_class = int(
        transaction["Class"].iloc[0]
    )

    if actual_class == 1:
        st.warning("⚠️ Actual dataset label: FRAUD")
    else:
        st.success("✅ Actual dataset label: LEGITIMATE")

    st.divider()

    if st.button(
        "🔍 Analyze Selected Transaction",
        use_container_width=True
    ):

        transaction_features = transaction[
            feature_names
        ]

        transaction_scaled = scaler.transform(
            transaction_features
        )

        prediction = model.predict(
            transaction_scaled
        )[0]

        probability = model.predict_proba(
            transaction_scaled
        )[0][1]

        st.subheader("🤖 Model Prediction")

        col1, col2 = st.columns(2)

        with col1:

            if prediction == 1:
                st.error(
                    "🚨 MODEL: FRAUD"
                )
            else:
                st.success(
                    "✅ MODEL: LEGITIMATE"
                )

        with col2:

            st.metric(
                "Fraud Probability",
                f"{probability * 100:.2f}%"
            )

        st.divider()

        # Compare prediction with actual label
        if prediction == actual_class:

            st.success(
                "🎯 Model prediction matches the actual dataset label!"
            )

        else:

            st.warning(
                "⚠️ Model prediction differs from the actual dataset label."
            )

# ============================================================
# FOOTER
# ============================================================
st.divider()

st.caption(
    "Fraud Detection AI | Python • Pandas • Scikit-learn • Streamlit"
)

