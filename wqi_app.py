import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# WQI Parameters (all numeric values as floats for consistency)
PARAMETERS = {
    'pH': {'V0': 7.0, 'Sn': 8.5, 'Wi': 0.10, 'Unit': ''},
    'DO': {'V0': 5.0, 'Sn': 14.6, 'Wi': 0.17, 'Unit': 'mg/L'},
    'BOD5': {'V0': 0.0, 'Sn': 3.0, 'Wi': 0.12, 'Unit': 'mg/L'},
    'Turbidity': {'V0': 0.0, 'Sn': 5.0, 'Wi': 0.08, 'Unit': 'NTU'},
    'Total_Coliform': {'V0': 0.0, 'Sn': 5.0, 'Wi': 0.15, 'Unit': 'CFU/100mL'},
    'Nitrate': {'V0': 0.0, 'Sn': 45.0, 'Wi': 0.10, 'Unit': 'mg/L'},
    'Residual_Chlorine': {'V0': 0.2, 'Sn': 4.0, 'Wi': 0.15, 'Unit': 'mg/L'},
    'Temp_Change': {'V0': 0.0, 'Sn': 3.0, 'Wi': 0.08, 'Unit': '°C'},
    'TDS': {'V0': 0.0, 'Sn': 500.0, 'Wi': 0.05, 'Unit': 'mg/L'}
}

def sub_index_qi(value, param):
    """Calculate sub-index Qi for a parameter (linear approximation)."""
    config = PARAMETERS[param]
    V0, Sn = config['V0'], config['Sn']
    
    if not isinstance(value, (int, float)) or pd.isna(value):
        return None
    if value > Sn:
        return 0.0
    elif param in ['BOD5', 'Turbidity', 'Nitrate', 'TDS', 'Temp_Change']:
        return 100.0 * ((Sn - value) / Sn)
    elif param == 'Total_Coliform':
        return 100.0 * (1.0 - np.log10(max(value, 1.0)) / np.log10(config['Sn'])) if value > 0 else 100.0
    else:
        return 100.0 * ((value - V0) / (Sn - V0)) if Sn != V0 else 100.0

def calculate_wqi(data):
    """Calculate WQI for a single sample."""
    qi_values = []
    weights = []
    
    for param in PARAMETERS:
        if param in data and data[param] is not None:
            qi = sub_index_qi(data[param], param)
            if qi is not None:
                qi_values.append(qi * PARAMETERS[param]['Wi'])
                weights.append(PARAMETERS[param]['Wi'])
    
    if not weights:
        return None, "No valid data provided"
    wqi = sum(qi_values) / sum(weights)
    
    # Classify WQI
    if wqi > 90: return wqi, 'Excellent'
    elif wqi > 70: return wqi, 'Good'
    elif wqi > 50: return wqi, 'Medium'
    elif wqi > 25: return wqi, 'Poor'
    else: return wqi, 'Very Poor'

# Streamlit App
st.set_page_config(page_title="WQI Calculator", page_icon="💧")
st.title("💧 Water Quality Index (WQI) Calculator")
st.markdown("""
This tool calculates the Water Quality Index (WQI) for drinking water samples based on key parameters.
Enter values for a single sample or upload a CSV for batch processing. Results include WQI scores and classifications.
""")

# Sidebar for Help
with st.sidebar:
    st.header("Help")
    st.info("""
    - **Input Guidance**: Enter values within typical ranges (e.g., pH 6.5–8.5, Turbidity ≤ 5 NTU).
    - **Standards**: Based on WHO/EPA guidelines.
    - **CSV Format**: Columns should include 'Location' and parameter names (e.g., pH, DO, BOD5).
    - **WQI Classes**: Excellent (>90), Good (>70), Medium (>50), Poor (>25), Very Poor (≤25).
    For support, contact your water quality team.
    """)

# Single Sample Input
st.header("Single Sample Input")
location = st.text_input("Sample Location (e.g., Plant A)", "Sample_001")

# Input form with float-only parameters
data = {}
for param, config in PARAMETERS.items():
    data[param] = st.number_input(
        f"{param} ({config['Unit']})",
        min_value=0.0 if param != 'pH' else 0.0,
        max_value=float(config['Sn'] * 1.5),  # Ensure float
        value=float(config['V0']),           # Ensure float
        step=0.1,                            # Float step
        help=f"Ideal: {config['V0']}, Max Standard: {config['Sn']} {config['Unit']}"
    )

# Calculate Button
if st.button("Calculate WQI"):
    wqi, wqi_class = calculate_wqi(data)
    if wqi is None:
        st.error(wqi_class)
    else:
        st.success(f"**WQI**: {wqi:.2f} (**{wqi_class}**) for {location}")
        
        # Visualize
        st.subheader("WQI Visualization")
        fig, ax = plt.subplots(figsize=(4, 6))
        sns.barplot(x=[location], y=[wqi], hue=[wqi_class], ax=ax)
        ax.set_ylim(0, 100)
        ax.set_ylabel("WQI Score")
        st.pyplot(fig)
        
        # Save to CSV
        result = pd.DataFrame({
            'Location': [location],
            'WQI': [wqi],
            'Class': [wqi_class],
            **data
        })
        csv = result.to_csv(index=False)
        st.download_button("Download Results", csv, f"wqi_{location}.csv", "text/csv")

# Batch Processing
st.header("Batch Processing (CSV Upload)")
st.markdown("Upload a CSV with columns: Location, pH, DO, BOD5, etc.")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if 'Location' not in df.columns:
        df['Location'] = [f"Sample_{i+1}" for i in range(len(df))]
    
    result_df = df.copy()
    result_df[['WQI', 'Class']] = result_df.apply(
        lambda row: calculate_wqi(row), axis=1, result_type='expand'
    )
    
    st.write("**Batch Results**", result_df[['Location', 'WQI', 'Class'] + list(PARAMETERS.keys())])
    
    # Visualize
    st.subheader("Batch WQI Visualization")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=result_df, x='Location', y='WQI', hue='Class', ax=ax)
    ax.set_ylim(0, 100)
    ax.set_ylabel("WQI Score")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Download
    csv = result_df.to_csv(index=False)
    st.download_button("Download Batch Results", csv, "wqi_batch_results.csv", "text/csv")
