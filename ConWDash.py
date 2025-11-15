# ConWDash.py — Churn Model Comparator Dashboard
import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import os

# Page config
st.set_page_config(page_title="Churn Model Comparator", layout="wide")
st.title("🤖 Churn Prediction: Supervised vs Unsupervised")

# Sidebar: Choose model (only churn models)
st.sidebar.header("⚙️ Select Model Type")

# Define expected model files
expected_models = {
    "Supervised Churn (Random Forest)": "churn_model.pkl",
    "Unsupervised Churn (Isolation Forest)": "churn_unsupervised_model.pkl"
}

selected_label = st.sidebar.selectbox("Choose Model", list(expected_models.keys()))
model_filename = expected_models[selected_label]

# Fixed dataset (only one)
data_filename = "churndata.csv"

# Paths
model_path = os.path.join('pkl', model_filename)
data_path = os.path.join('data', data_filename)

# Validation
if not os.path.exists('pkl') or not os.path.exists('data'):
    st.error("❌ Required folders 'pkl' or 'data' missing!")
    st.stop()

if not os.path.exists(model_path):
    st.error(f"❌ Model file not found: {model_path}")
    st.stop()

if not os.path.exists(data_path):
    st.error(f"❌ Dataset not found: {data_path}")
    st.stop()

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()
    return df

# Load model(s)
@st.cache_resource
def load_supervised_model():
    return joblib.load(os.path.join('pkl', 'churn_model.pkl'))

@st.cache_resource
def load_unsupervised_components():
    model = joblib.load(os.path.join('pkl', 'churn_unsupervised_model.pkl'))
    preprocessor = joblib.load(os.path.join('pkl', 'churn_unsupervised_preprocessor.pkl'))
    return model, preprocessor

# Load everything
df = load_data()
y_true = (df['Churn'] == 'Yes').astype(int)

if "Supervised" in selected_label:
    model = load_supervised_model()
    X = df.drop(['customerID', 'Churn'], axis=1)
    y_pred = model.predict(X)
    
    # Get probabilities for ranking
    churn_prob = model.predict_proba(X)[:, 1]
    df['Churn_Prob'] = churn_prob

else:  # Unsupervised
    model, preprocessor = load_unsupervised_components()
    X_raw = df.drop(['customerID', 'Churn'], axis=1)
    X_processed = preprocessor.transform(X_raw)
    
    # Predict: -1 = anomaly (churn), 1 = normal
    y_pred_outlier = model.predict(X_processed)
    y_pred = [1 if x == -1 else 0 for x in y_pred_outlier]
    
    # Get anomaly scores for ranking
    anomaly_scores = model.decision_function(X_processed)
    df['Churn_Prob'] = -anomaly_scores  # Higher = more likely to churn

# Display results
st.subheader(f"📊 {selected_label} — Performance Metrics")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
st.write("Confusion Matrix:")
st.write(cm)

# Classification Report
report = classification_report(y_true, y_pred, target_names=['Not Churn', 'Churn'], output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.write("Classification Report:")
st.dataframe(report_df.style.format("{:.2f}"))

# Top High-Risk Customers
st.subheader("⚠️ Top 10 Customers at Highest Risk of Churning")
high_risk = df.nlargest(10, 'Churn_Prob')[['customerID', 'Churn_Prob', 'MonthlyCharges', 'tenure']]
st.dataframe(high_risk.style.format({"Churn_Prob": "{:.1%}"}))

# Footer
st.markdown("---")
st.caption(f"Model: {selected_label} | Dataset: {data_filename} | Updated: Real-time")