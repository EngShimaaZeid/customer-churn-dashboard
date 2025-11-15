# customer-churn-dashboard
Customer Churn Prediction Dashboard — Compare supervised (Random Forest) and unsupervised (Isolation Forest) ML models on real telecom data. Interactive Streamlit dashboard with risk profiling, performance metrics (Recall, Precision), and MLOps best practices. Built for “forecasting &amp; anomaly models” job scope. 

machine-learning | churn-prediction | streamlit | anomaly-detection | mlops | python

# Customer Churn Prediction Dashboard  
**Supervised vs. Unsupervised Machine Learning Comparison**

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

A comparative machine learning dashboard that evaluates **supervised** and **unsupervised** approaches to predict customer churn using real-world telecom data. Built to fulfill core requirements of modern ML engineering roles: **model development**, **dashboard integration**, **performance validation**, and **reproducible workflows**.

## 🎯 Project Overview
This project demonstrates end-to-end implementation of two distinct ML strategies on the **Telco Customer Churn dataset**:
- **Supervised Learning**: `RandomForestClassifier` trained with labeled churn data.
- **Unsupervised Learning**: `IsolationForest` treating churn as an anomaly (no labels used during training).

Both models are evaluated, compared, and visualized in an interactive **Streamlit dashboard**, enabling stakeholders to:
- View model performance metrics (Recall, Precision, F1-Score)
- Identify top customers at risk of churning
- Understand trade-offs between modeling approaches

## 📦 Features
- ✅ **Dual-model comparison** on the same dataset
- ✅ **Interactive dashboard** with real-time risk profiling
- ✅ **Robust preprocessing pipeline** (scaling, encoding, cleaning)
- ✅ **Model persistence** for deployment-ready workflows
- ✅ **Performance validation** using business-relevant metrics (focus on **Recall**)

## 🛠️ Technologies Used
| Category          | Tools & Libraries                          |
|-------------------|--------------------------------------------|
| **Core ML**       | scikit-learn, pandas, numpy                |
| **Dashboard**     | Streamlit                                  |
| **Model Types**   | Random Forest (Supervised), Isolation Forest (Unsupervised) |
| **Preprocessing** | StandardScaler, OneHotEncoder, ColumnTransformer |
| **Evaluation**    | Confusion Matrix, Classification Report    |
| **Deployment**    | joblib (model serialization)               |

## 📂 Project Structure
