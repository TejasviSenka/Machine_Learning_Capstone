import streamlit as st
import pandas as pd

# Set page config
st.set_page_config(page_title="ML Capstone Dashboard", layout="wide")

st.title("☀️ Solar Power Generation Dashboard")
st.markdown("This is a minimal web interface for the Machine Learning Capstone project. Later, we can load our `.pkl` models here for live predictions!")

import os

# Cache the dataset loading so it's fast
@st.cache_data
def load_data():
    # Dynamically build path so it works regardless of where the app is run from
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "..", "data", "solar_generation.csv")
    return pd.read_csv(csv_path)

try:
    df = load_data()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
    with col2:
        st.subheader("AC Power vs Irradiance")
        # A simple scatter chart using Streamlit's native charting
        st.scatter_chart(data=df, x='IRR (W/m2)', y='AC Power in Watts')
        
except FileNotFoundError:
    st.error("⚠️ Dataset not found. Make sure `solar_generation.csv` is inside the `data/` folder.")
