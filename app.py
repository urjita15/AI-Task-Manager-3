
import os, json, joblib, pandas as pd, numpy as np
import streamlit as st
from datetime import datetime
from scipy.sparse import hstack, csr_matrix
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="AI Task Management Dashboard", layout="wide")

ARTIFACT_DIR = "artifacts"

@st.cache_resource
def load_artifacts():
    tfidf = joblib.load(os.path.join(ARTIFACT_DIR, "tfidf_vectorizer.joblib"))
    nb = joblib.load(os.path.join(ARTIFACT_DIR, "nb_model.joblib"))
    svm = joblib.load(os.path.join(ARTIFACT_DIR, "svm_model.joblib"))
    lr = joblib.load(os.path.join(ARTIFACT_DIR, "lr_model.joblib"))
    rf = joblib.load(os.path.join(ARTIFACT_DIR, "rf_model.joblib"))
    xgb = joblib.load(os.path.join(ARTIFACT_DIR, "xgb_model.joblib"))
    cat_le = joblib.load(os.path.join(ARTIFACT_DIR, "cat_label_encoder.joblib"))
    pri_le = joblib.load(os.path.join(ARTIFACT_DIR, "pri_label_encoder.joblib"))
    return tfidf, nb, svm, lr, rf, xgb, cat_le, pri_le

@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(ARTIFACT_DIR, "tasks_synthetic.csv"))
    return df

tfidf, nb, svm, lr, rf, xgb, cat_le, pri_le = load_artifacts()
df = load_data()

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", [
    "Add New Task", "View All Tasks", "Workload Analyzer", "Prioritize & Manage", "Insights / Models"
])

if page == "Add New Task":
    st.title("Add New Task")
    desc = st.text_area("Task Description", "")
    days_left = st.number_input("Days Left", 0, 30, 5)
    user_workload = st.number_input("User Workload", 0, 10, 3)
    task_length = len(desc.split())

    if st.button("Predict"):
        if desc.strip():
            X_text = tfidf.transform([desc])
            X_num = csr_matrix([[days_left, task_length, user_workload]])
            X = hstack([X_text, X_num])
            cat = cat_le.inverse_transform(svm.predict(X))[0]
            pri = pri_le.inverse_transform(rf.predict(X))[0]
            st.success(f"Predicted Category: {cat}")
            st.info(f"Predicted Priority: {pri}")
        else:
            st.warning("Enter a task description first.")

elif page == "View All Tasks":
    st.title("All Tasks")
    st.dataframe(df, use_container_width=True)

elif page == "Workload Analyzer":
    st.title("Workload Analyzer")
    workload = df.groupby("assigned_user")["user_workload"].sum().reset_index()
    st.plotly_chart(px.bar(workload, x="assigned_user", y="user_workload", title="Workload by User"))

elif page == "Prioritize & Manage":
    st.title("Prioritize & Manage")
    df_sorted = df.sort_values("priority", ascending=True)
    st.dataframe(df_sorted)

elif page == "Insights / Models":
    st.title("Model Insights / Performance")
    metrics_df = pd.read_csv(os.path.join(ARTIFACT_DIR, "metrics_report.csv"))
    st.dataframe(metrics_df)
    st.plotly_chart(px.bar(metrics_df, x=metrics_df.index, y="Accuracy", title="Model Accuracy"))
