import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import ks_2samp
import os
import numpy as np

# Set page config
st.set_page_config(page_title="Credit Scoring Monitoring", layout="wide")

st.title("Credit Scoring Model Monitoring")

# Load Production Logs
LOG_FILE = "production_logs.json"

@st.cache_data
def load_production_data():
    if not os.path.exists(LOG_FILE):
        return pd.DataFrame()
    
    with open(LOG_FILE, 'r') as f:
        try:
            logs = json.load(f)
        except json.JSONDecodeError:
            return pd.DataFrame()
            
    if not logs:
        return pd.DataFrame()
        
    # Extract input data and predictions
    data = []
    for entry in logs:
        if entry.get("status") == "success":
            row = entry.get("input", {}).copy()
            row["timestamp"] = entry.get("timestamp")
            row["prediction"] = entry.get("prediction", {}).get("prediction")
            row["probability"] = entry.get("prediction", {}).get("probability")
            data.append(row)
            
    return pd.DataFrame(data)

# Load Reference Data
REFERENCE_FILE = "data/processed/train_prepared.csv"

@st.cache_data
def load_reference_data():
    if not os.path.exists(REFERENCE_FILE):
        return pd.DataFrame()
    # Load separate sample to avoid memory issues if large
    return pd.read_csv(REFERENCE_FILE, nrows=10000)

prod_df = load_production_data()
ref_df = load_reference_data()

# Dashboard Layout
tab1, tab2 = st.tabs(["API Performance", "Data Drift Analysis"])

with tab1:
    st.header("API Performance & Usage")
    
    if prod_df.empty:
        st.warning("No production data available yet.")
    else:
        # Convert timestamp
        prod_df["timestamp"] = pd.to_datetime(prod_df["timestamp"])
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Requests", len(prod_df))
        col2.metric("Positive Predictions (1)", prod_df["prediction"].sum())
        col3.metric("Defaut Rate (Prod)", f"{prod_df['prediction'].mean():.2%}")
        
        # Timeline
        requests_over_time = prod_df.set_index("timestamp").resample('H').size()
        st.subheader("Requests over Time")
        st.line_chart(requests_over_time)
        
        # Prediction Distribution
        st.subheader("Prediction Probability Distribution")
        fig = px.histogram(prod_df, x="probability", nbins=20, title="Production Probability Distribution")
        st.plotly_chart(fig)

with tab2:
    st.header("Data Drift Analysis")
    
    if prod_df.empty or ref_df.empty:
        st.warning("Need both production logs and reference data for drift analysis.")
        if ref_df.empty:
            st.error(f"Reference data not found at {REFERENCE_FILE}")
    else:
        st.markdown("Comparing **Reference Data** (Training Sample) vs **Production Data** (Live Requests).")
        
        # Select common numerical columns (excluding IDs, etc)
        ignore_cols = ["SK_ID_CURR", "TARGET", "timestamp", "prediction", "probability"]
        numeric_cols = [c for c in prod_df.columns if c in ref_df.columns and pd.api.types.is_numeric_dtype(prod_df[c]) and c not in ignore_cols]
        
        selected_feature = st.selectbox("Select Feature to Analyze", numeric_cols)
        
        if selected_feature:
            # Drift Metric (KS Test)
            try:
                # Handle NaNs
                ref_data = ref_df[selected_feature].dropna()
                prod_data = prod_df[selected_feature].dropna() # Explicit conversion not needed if metrics check passed
                
                # Check for incompatible types (e.g. string vs float) caused by JSON loading
                prod_data = pd.to_numeric(prod_data, errors='coerce').dropna()
                
                if len(prod_data) < 5:
                    st.info("Not enough production data for statistical drift test.")
                else:
                    ks_stat, p_value = ks_2samp(ref_data, prod_data)
                    
                    col1, col2 = st.columns(2)
                    col1.metric("KS Statistic", f"{ks_stat:.4f}")
                    col2.metric("P-Value", f"{p_value:.4f}")
                    
                    if p_value < 0.05:
                        st.error("Significant Drift Detected! (p-value < 0.05)")
                    else:
                        st.success("No Significant Drift Detected.")
                    
                    # Visual Comparison
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(x=ref_data, name='Reference', opacity=0.5, histnorm='probability density'))
                    fig.add_trace(go.Histogram(x=prod_data, name='Production', opacity=0.5, histnorm='probability density'))
                    
                    fig.update_layout(title=f"Distribution Comparison: {selected_feature}", barmode='overlay')
                    st.plotly_chart(fig)
                    
            except Exception as e:
                st.error(f"Error analyzing feature {selected_feature}: {e}")

