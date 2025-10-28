
import streamlit as st
import pandas as pd
import joblib
from scipy.sparse import hstack, csr_matrix
import numpy as np

st.set_page_config(page_title="AI Task Manager Dashboard", layout="wide")

@st.cache_resource
def load_artifacts():
    tfidf = joblib.load("artifacts/tfidf_vectorizer.joblib")
    svm = joblib.load("artifacts/svm_model.joblib")
    rf = joblib.load("artifacts/rf_model.joblib")
    cat_le = joblib.load("artifacts/cat_label_encoder.joblib")
    pri_le = joblib.load("artifacts/pri_label_encoder.joblib")
    return tfidf, svm, rf, cat_le, pri_le

tfidf, svm, rf, cat_le, pri_le = load_artifacts()

st.sidebar.title("Pages")
page = st.sidebar.radio(" ", [
    "Add New Task",
    "View All Tasks",
    "Workload Analyzer",
    "Prioritize & Manage",
    "Insights / Models"
])

if page == "Add New Task":
    st.title("📝 Add New Task")
    desc = st.text_area("Task Description")
    days_left = st.number_input("Days Left", 0, 30, 5)
    user_workload = st.number_input("User Workload", 0, 10, 3)
    task_length = len(desc.split())
    if st.button("Predict"):
        X_text = tfidf.transform([desc])
        X_num = csr_matrix([[days_left, task_length, user_workload]])
        X = hstack([X_text, X_num])
        cat_pred = cat_le.inverse_transform(svm.predict(X))[0]
        pri_pred = pri_le.inverse_transform(rf.predict(X))[0]
        st.success(f"Predicted Category: {cat_pred}")
        st.info(f"Predicted Priority: {pri_pred}")

elif page == "View All Tasks":
    st.title("📋 View All Tasks")
    try:
        df = pd.read_csv("artifacts/tasks_synthetic.csv")
        st.dataframe(df)
    except FileNotFoundError:
        st.error("Task dataset not found. Please ensure 'tasks_synthetic.csv' exists in /artifacts.")

elif page == "Workload Analyzer":
    st.title("📊 Workload Analyzer")
    try:
        df = pd.read_csv("artifacts/tasks_synthetic.csv")
        workload_summary = df.groupby("Assigned User")["Workload"].mean().reset_index()
        st.bar_chart(workload_summary.set_index("Assigned User"))
    except Exception:
        st.warning("Data unavailable for workload analysis.")

elif page == "Prioritize & Manage":
    st.title("⚙️ Prioritize & Manage")
    try:
        df = pd.read_csv("artifacts/tasks_synthetic.csv")
        df_sorted = df.sort_values(by="Priority", ascending=False)
        st.dataframe(df_sorted)
    except Exception:
        st.warning("Unable to load task data for prioritization.")

elif page == "Insights / Models":
    st.title("📈 Insights & Model Performance")
    try:
        metrics = joblib.load("artifacts/metrics_report.joblib")
        st.json(metrics)
    except Exception:
        st.info("Metrics report not found. Please ensure model training artifacts exist.")
