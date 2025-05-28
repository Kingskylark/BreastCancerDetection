import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon="assets/breastCancer.jpeg",
    layout="centered"
)

# Load model and scaler
model = joblib.load("naive_bayes_model.pkl")
scaler = joblib.load("scaler.pkl")

# Features list
features = [
    'concave points_worst', 'perimeter_worst', 'concave points_mean',
    'radius_worst', 'perimeter_mean', 'area_worst',
    'radius_mean', 'area_mean', 'concavity_mean', 'concavity_worst'
]

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "Home"

# Navigation function
def set_page(p):
    st.session_state.page = p


def render_navbar():
    st.markdown("""
    <style>
        .nav-container {
            background-color: #f3f3f3;
            padding: 10px 20px;
            border-radius: 12px;
            box-shadow: 0px 2px 6px rgba(0, 0, 0, 0.1);
            display: flex;
            justify-content: center;
            margin-bottom: 25px;
        }
        .nav-button {
            background-color: transparent;
            border: none;
            color: #333;
            padding: 10px 20px;
            margin: 0 10px;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
        }
        .nav-button.active {
            background-color: #6c63ff;
            color: white;
            border-radius: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        if st.button("🏠 Home", key="home"):
            set_page("Home")
    with col2:
        if st.button("✍️ Manual Input", key="manual"):
            set_page("Manual Input")
    with col3:
        if st.button("📁 Upload", key="upload"):
            set_page("Upload Dataset")
    with col4:
        if st.button("ℹ️ About", key="about"):
            set_page("About / Class Info")

# Pages
def home():
    st.image("assets/breastCancer.jpeg", width=100)
    st.title("Breast Cancer Prediction System")
    st.markdown("""
    Welcome to the **Breast Cancer Prediction System**.

    This tool allows you to either **upload a dataset** for batch predictions, or **manually enter feature values** to predict whether a breast tumor is **Benign** or **Malignant**.
    """)
    st.info("Use the navbar above to get started.")

def manual_input():
    st.header("Manual Input Prediction")
    patient_name = st.text_input("Enter Patient Name")
    st.subheader("Input Features")
    user_input = []

    for feature in features:
        if feature in ['area_mean', 'area_worst']:
            value = st.number_input(f"{feature}", min_value=0.0, max_value=5000.0, step=1.0)
        elif feature in ['perimeter_mean', 'perimeter_worst']:
            value = st.number_input(f"{feature}", min_value=0.0, max_value=1000.0, step=0.5)
        else:
            value = st.slider(f"{feature}", min_value=0.0, max_value=100.0, step=0.1)
        user_input.append(value)

    if st.button("Predict for Patient"):
        input_array = np.array(user_input).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]

        result = "Malignant" if prediction == 1 else "Benign"
        confidence = round(np.max(probabilities) * 100, 2)

        st.success(f"Prediction for {patient_name or 'Unnamed Patient'}: {result}")
        st.info(f"Confidence: {confidence}%")

        fig, ax = plt.subplots()
        sns.barplot(x=["Benign", "Malignant"], y=probabilities, ax=ax)
        ax.set_title("Prediction Probability")
        st.pyplot(fig)

def upload_dataset():
    st.header("Batch Prediction from CSV")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        missing_cols = [f for f in features if f not in data.columns]
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
        else:
            input_scaled = scaler.transform(data[features])
            predictions = model.predict(input_scaled)
            probabilities = model.predict_proba(input_scaled)
            results = pd.DataFrame({
                "Prediction": ["Malignant" if p == 1 else "Benign" for p in predictions],
                "Confidence (%)": [round(np.max(prob, initial=0)*100, 2) for prob in probabilities]
            })
            output = pd.concat([data, results], axis=1)
            st.dataframe(output)

            st.subheader("Select a Patient for Detailed Analysis")
            patient_index = st.selectbox("Select Row Index", options=output.index.tolist())
            patient_data = output.loc[patient_index]
            st.write(f"Selected Patient Prediction: {patient_data['Prediction']}")
            st.write(f"Confidence: {patient_data['Confidence (%)']}%")

            fig, ax = plt.subplots()
            sns.barplot(x=features, y=patient_data[features].values, ax=ax)
            ax.set_title("Feature Values for Selected Patient")
            ax.set_ylabel("Value")
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)

            csv = output.to_csv(index=False).encode("utf-8")
            st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")

def about():
    st.header("About & Class Information")
    st.markdown("""
    **Prediction Classes:**
    - **Benign (0):** Non-cancerous tumor
    - **Malignant (1):** Cancerous tumor, potentially life-threatening

    **Model Used:** Gaussian Naive Bayes
    
    **Features Explained:**
    These features are derived from digitized images of breast mass FNA scans.
    """)

# Page mapping
pages = {
    "Home": home,
    "Manual Input": manual_input,
    "Upload Dataset": upload_dataset,
    "About / Class Info": about
}

# Render navigation bar
render_navbar()

# Display selected page
pages[st.session_state.page]()
