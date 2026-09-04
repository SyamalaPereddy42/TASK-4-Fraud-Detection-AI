\# 💳 Fraud Detection AI



An end-to-end Machine Learning project that detects potentially fraudulent credit card transactions using \*\*Logistic Regression\*\* and provides an interactive \*\*Streamlit dashboard\*\* for prediction and model evaluation.



\## 📌 Project Overview



Credit card fraud detection is a highly imbalanced classification problem where fraudulent transactions are much fewer than legitimate transactions.



This project uses Machine Learning to classify transactions as:



\* ✅ \*\*Legitimate\*\*

\* 🚨 \*\*Fraudulent\*\*



The project includes model training, evaluation, transaction prediction, and an interactive Streamlit web application.



\## 🎯 Problem Statement



Build a Machine Learning system capable of identifying fraudulent credit card transactions while maintaining high fraud detection recall.



Because missing a fraudulent transaction can be costly, \*\*Fraud Recall\*\* is an important evaluation metric for this project.



\## 📊 Dataset



The project uses the \*\*Credit Card Fraud Detection dataset\*\* containing:



\* \*\*284,807 transactions\*\*

\* \*\*492 fraudulent transactions\*\*

\* \*\*284,315 legitimate transactions\*\*

\* 30 input features

\* Target variable: `Class`



> ⚠️ The `creditcard.csv` dataset is not included in this repository because the file is larger than GitHub's standard 100 MB individual-file limit.



To run the project locally, place the dataset at:



```text

Fraud\_Detection/data/creditcard.csv

```



\## 🤖 Machine Learning Model



The project uses:



\*\*Logistic Regression\*\*



Important techniques used:



\* Train/test split with stratification

\* Feature scaling using `StandardScaler`

\* `class\_weight="balanced"` to handle class imbalance

\* Model evaluation using Accuracy, Precision, Recall, F1-score and ROC-AUC

\* Joblib for saving the trained model



\## 📈 Model Performance



| Metric         | Result      |

| -------------- | ----------- |

| Accuracy       | \*\*97.55%\*\*  |

| Fraud Recall   | \*\*92%\*\*     |

| ROC-AUC        | \*\*97.21%\*\*  |

| Fraud Detected | \*\*90 / 98\*\* |



\### Confusion Matrix



```text

&#x20;                Predicted

&#x20;             Legit   Fraud

Actual Legit  55478    1386

Actual Fraud      8      90

```



The model detected \*\*90 out of 98 fraudulent transactions\*\* in the test set.



Because the dataset is highly imbalanced, accuracy should not be considered alone. Fraud recall is particularly important for this application.



\## 🖥️ Streamlit Application



The Streamlit dashboard provides:



\### 🏠 Dashboard



\* Total transaction count

\* Fraud transaction count

\* Legitimate transaction count

\* Fraud percentage

\* Dataset preview

\* Transaction distribution chart



\### 🔍 Transaction Prediction



Users can enter transaction feature values and receive:



\* Fraud / Legitimate prediction

\* Fraud probability

\* Visual prediction result



\### 📊 Model Performance



Displays:



\* Accuracy

\* Fraud Recall

\* ROC-AUC

\* Fraud detection statistics

\* Confusion matrix



\### 🧪 Test Real Transaction



Users can select an actual transaction from the dataset and compare:



\* Actual label

\* Model prediction

\* Fraud probability



\## 🛠️ Technologies Used



\* Python

\* Pandas

\* NumPy

\* Scikit-learn

\* Joblib

\* Streamlit

\* Matplotlib

\* Seaborn



\## 📂 Project Structure



```text

Fraud\_Detection/

│

├── app.py

├── train\_model.py

├── .gitignore

│

├── data/

│   └── creditcard.csv

│

└── models/

&#x20;   ├── fraud\_model.pkl

&#x20;   ├── scaler.pkl

&#x20;   └── feature\_names.pkl

```



\## ⚙️ Installation \& Setup



\### 1. Clone the repository



```bash

git clone https://github.com/SyamalaPereddy42/TASK-AIML-Training.git

```



\### 2. Navigate to the Fraud Detection project



```bash

cd TASK-AIML-Training/Fraud\_Detection

```



\### 3. Install required packages



```bash

pip install pandas numpy scikit-learn joblib streamlit matplotlib seaborn

```



\### 4. Add the dataset



Download the Credit Card Fraud Detection dataset and place:



```text

creditcard.csv

```



inside:



```text

Fraud\_Detection/data/

```



\### 5. Train the model



```bash

python train\_model.py

```



This generates the trained model files inside the `models` folder.



\### 6. Run the Streamlit application



```bash

python -m streamlit run app.py

```



Then open the local Streamlit URL displayed in the terminal.



\## 🔮 Future Improvements



\* Try advanced models such as Random Forest, XGBoost and neural networks

\* Apply SMOTE or other advanced imbalance-handling techniques

\* Perform hyperparameter optimization

\* Add precision-recall curves

\* Add transaction-level explainability using SHAP

\* Deploy the application to a cloud platform

\* Add real-time transaction monitoring



\## 👩‍💻 Author



\*\*Syamala Pereddy\*\*



B.Tech – Electronics and Communication Engineering



\### Project Focus



\*\*Machine Learning | Fraud Detection | Python | Scikit-learn | Streamlit\*\*



