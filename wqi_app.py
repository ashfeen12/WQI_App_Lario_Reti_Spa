import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Model Configurations ---

# Parameters for Springs
SPRINGS_PARAMS = {
    'Aluminum [µg/l Al]': {'weight': 0.001828, 'standard': 200},
    'Ammonium [mg/l NH4]': {'weight': 0.040472, 'standard': 0.5},
    'Arsenic [µg/l As]': {'weight': 0.009174, 'standard': 10},
    'Cadmium [µg/l Cd]': {'weight': 0.033676, 'standard': 5},
    'Chlorides [mg/l Cl]': {'weight': 0.038744, 'standard': 250},
    'Chlorites [µg/l ClO2]': {'weight': 0.018832, 'standard': 700},
    'Copper [mg/l Cu]': {'weight': 0.119593, 'standard': 2},
    'Fluorides [mg/l F]': {'weight': 0.01348, 'standard': 1.5},
    'Hardness [°F]': {'weight': 0.100148, 'standard': 50},
    'Iron [µg/l Fe]': {'weight': 0.040982, 'standard': 200},
    'Lead [µg/l Pb]': {'weight': 0.096853, 'standard': 10},
    'Magnesium [mg/l Mg]': {'weight': 0.111423, 'standard': 30},
    'Manganese [µg/l Mn]': {'weight': 0.04212, 'standard': 50},
    'Nitrates [mg/l NO3]': {'weight': 0.019065, 'standard': 50},
    'pH': {'weight': 0.076599, 'low': 6.5, 'high': 9.5},
    'Sodium [mg/l Na]': {'weight': 0.016428, 'standard': 200},
    'Sulfates [mg/l SO4]': {'weight': 0.078955, 'standard': 250},
    'Turbidity [NTU]': {'weight': 0.049485, 'standard': 0.3},
    'Vanadium [µg/l V]': {'weight': 0.010561, 'standard': 140},
    'Zinc [µg/l Zn]': {'weight': 0.081582, 'standard': 5000}
}

# Parameters for Wells
WELLS_PARAMS = {
    'Aluminum [µg/l Al]': {'weight': 0.08524457, 'standard': 200},
    'Ammonium [mg/l NH4]': {'weight': 0.00472345, 'standard': 0.5},
    'Arsenic [µg/l As]': {'weight': 0.02818904, 'standard': 10},
    'Cadmium [µg/l Cd]': {'weight': 0.07304108, 'standard': 5},
    'Chlorides [mg/l Cl]': {'weight': 0.05636716, 'standard': 250},
    'Copper [mg/l Cu]': {'weight': 0.0546783, 'standard': 2},
    'Fluorides [mg/l F]': {'weight': 0.08794153, 'standard': 1.5},
    'Hardness [°F]': {'weight': 0.05360424, 'standard': 50},
    'Iron [µg/l Fe]': {'weight': 0.07653616, 'standard': 200},
    'Lead [µg/l Pb]': {'weight': 0.15238569, 'standard': 10},
    'Magnesium [mg/l Mg]': {'weight': 0.06770434, 'standard': 30},
    'Manganese [µg/l Mn]': {'weight': 0.04199309, 'standard': 50},
    'Nitrates [mg/l NO3]': {'weight': 0.02516202, 'standard': 50},
    'pH': {'weight': 0.03543887, 'low': 6.5, 'high': 9.5},
    'Sodium [mg/l Na]': {'weight': 0.07688634, 'standard': 200},
    'Sulfates [mg/l SO4]': {'weight': 0.04959354, 'standard': 250},
    'Turbidity [NTU]': {'weight': 0.0030536, 'standard': 0.3},
    'Vanadium [µg/l V]': {'weight': 0.00286481, 'standard': 140},
    'Zinc [µg/l Zn]': {'weight': 0.02459217, 'standard': 5000}
}

# Parameters for Lakes
LAKES_PARAMS = {
    'Aluminum [µg/l Al]': {'weight': 0.13732495, 'standard': 200},
    'Ammonium [mg/l NH4]': {'weight': 0.08499825, 'standard': 0.5},
    'Arsenic [µg/l As]': {'weight': 0.00785124, 'standard': 10},
    'Cadmium [µg/l Cd]': {'weight': 0.03023614, 'standard': 5},
    'Calcium [mg/l Ca]': {'weight': 0.0037585, 'standard': 300},
    'Chlorides [mg/l Cl]': {'weight': 0.12006373, 'standard': 250},
    'Conductivity at 20°C [µS/cm]': {'weight': 0.10301697, 'standard': 2500},
    'Copper [mg/l Cu]': {'weight': 0.05903513, 'standard': 2},
    'Fluorides [mg/l F]': {'weight': 0.05134627, 'standard': 1.5},
    'Hardness [°F]': {'weight': 0.00831767, 'standard': 50},
    'Iron [µg/l Fe]': {'weight': 0.02670118, 'standard': 200},
    'Lead [µg/l Pb]': {'weight': 0.02825393, 'standard': 10},
    'Manganese [µg/l Mn]': {'weight': 0.00192557, 'standard': 50},
    'Nitrates [mg/l NO3]': {'weight': 0.02986598, 'standard': 50},
    'pH': {'weight': 0.03950439, 'low': 6.5, 'high': 9.5},
    'Sodium [mg/l Na]': {'weight': 0.08998053, 'standard': 200},
    'Sulfates [mg/l SO4]': {'weight': 0.02081114, 'standard': 250},
    'Turbidity [NTU]': {'weight': 0.08237665, 'standard': 0.3},
    'Vanadium [µg/l V]': {'weight': 0.03977873, 'standard': 140},
    'Zinc [µg/l Zn]': {'weight': 0.03485305, 'standard': 5000}
}

# --- WQI Calculation Functions ---

def calculate_sub_index(value, param, param_config):
    """Calculates the sub-index for a given parameter."""
    if pd.isna(value):
        return None
    if param == 'pH':
        return abs((value - 7) / (param_config['high'] - 7)) * 100
    else:
        return (value / param_config['standard']) * 100

def calculate_wqi(data, params):
    """Calculates the Water Quality Index (WQI)."""
    wqi = 0
    total_weight = 0
    for param, config in params.items():
        if param in data and not pd.isna(data[param]):
            sub_index = calculate_sub_index(data[param], param, config)
            if sub_index is not None:
                wqi += sub_index * config['weight']
                total_weight += config['weight']
    
    if total_weight == 0:
        return None, "No valid data provided"
        
    return wqi, classify_wqi(wqi)

def classify_wqi(wqi):
    """Classifies the WQI score."""
    if wqi < 50:
        return 'Excellent water'
    elif wqi <= 100:
        return 'Good water'
    elif wqi <= 200:
        return 'Poor water'
    elif wqi <= 300:
        return 'Very poor water'
    else:
        return 'Unsuitable for drinking'

# --- Streamlit App ---

st.set_page_config(page_title="WQI Calculator", page_icon="💧", layout="wide")

# --- HEADER WITH MAIN LOGO AND TITLE ---
col1, col2 = st.columns([1, 4])
with col1:
    st.image("logolario.png", width=150) # Main company logo
with col2:
    st.title("Lario Reti Holding Spa - WQI Tool")
    st.markdown("A tool for calculating the Water Quality Index (WQI) for different water bodies.")


# Sidebar for Model Selection and Help
with st.sidebar:
    st.header("Settings")
    water_body = st.selectbox(
        "Select Water Body",
        ("Springs", "Wells", "Lakes")
    )
    st.header("Help")
    st.info("""
        - **Select a Water Body**: Choose between Springs, Wells, or Lakes.
        - **Single Sample**: Manually enter values.
        - **Batch Processing**: Upload a CSV with parameter columns.
        - **WQI Classification**:
            - < 50: Excellent
            - 50-100: Good
            - 100-200: Poor
            - 200-300: Very Poor
            - > 300: Unsuitable
    """)

# Parameter selection based on water body
if water_body == "Springs":
    PARAMETERS = SPRINGS_PARAMS
elif water_body == "Wells":
    PARAMETERS = WELLS_PARAMS
else:
    PARAMETERS = LAKES_PARAMS

# --- Main App Sections (using tabs) ---
tab1, tab2 = st.tabs(["Single Sample Input", "Batch Processing (CSV Upload)"])

with tab1:
    # ... (The content of this tab remains the same)
    st.header(f"Single Sample Input for {water_body}")
    
    with st.form(key='single_sample_form'):
        location = st.text_input("Sample Location (e.g., Plant A)", "Sample_001")
        
        cols = st.columns(3)
        data = {}
        col_index = 0
        for param, config in PARAMETERS.items():
            with cols[col_index % 3]:
                if param == 'pH':
                    data[param] = st.number_input(label=param, min_value=0.0, max_value=14.0, value=7.0, step=0.1)
                else:
                    data[param] = st.number_input(label=param, min_value=0.0, value=0.0, step=0.1)
            col_index += 1
            
        submit_button = st.form_submit_button(label='Calculate WQI')

    if submit_button:
        wqi, wqi_class = calculate_wqi(data, PARAMETERS)
        if wqi is None:
            st.error(wqi_class)
        else:
            st.success(f"**WQI for {location}**: {wqi:.2f} - **{wqi_class}**")
            
            st.subheader("WQI Visualization")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x=[location], y=[wqi], palette=["#2ecc71" if wqi < 50 else "#f1c40f" if wqi <= 100 else "#e74c3c" if wqi <= 300 else "#c0392b"])
            ax.set_ylim(0, max(350, wqi * 1.2))
            ax.set_ylabel("WQI Score")
            st.pyplot(fig)

            result_df = pd.DataFrame({'Location': [location], 'WQI': [wqi], 'Class': [wqi_class], **data})
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download Results as CSV", data=csv, file_name=f'wqi_{location}.csv', mime='text/csv')

with tab2:
    # ... (The content of this tab remains the same)
    st.header(f"Batch Processing for {water_body}")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        for param in PARAMETERS.keys():
            if param not in df.columns:
                df[param] = np.nan

        results = df.apply(lambda row: calculate_wqi(row, PARAMETERS), axis=1, result_type='expand')
        df[['WQI', 'Class']] = results

        st.write("### Batch Results")
        st.dataframe(df)

        st.write("### WQI Distribution for Batch")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=df, x=df.index, y='WQI', hue='Class', dodge=False)
        plt.xticks(rotation=45)
        ax.set_ylim(0, max(350, df['WQI'].max() * 1.2))
        st.pyplot(fig)
        
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download Batch Results as CSV", data=csv, file_name='wqi_batch_results.csv', mime='text/csv')

# --- FOOTER WITH PARTNER LOGOS ---
st.markdown("---")
st.subheader("Project Partners")
fcol1, fcol2, fcol3 = st.columns(3)
with fcol1:
    st.image("weblogoall.jpg")

