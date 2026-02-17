import streamlit as st
import numpy as np
import joblib
import base64
import os

# =============== PAGE CONFIG ===============
st.set_page_config(
    page_title="Sleep Disorder Prediction Using ML",
    page_icon="😴",
    layout="wide",
)

# =============== LOAD PIPELINE ===============
pipeline = joblib.load("sleep_pipeline2.pkl")

model = pipeline["model"]
scaler = pipeline["scaler"]
le_gender = pipeline["le_gender"]
le_occupation = pipeline["le_occupation"]
le_bmi = pipeline["le_bmi"]
le_sleep = pipeline["le_sleep"]
feature_cols = pipeline["feature_cols"]

# =============== HEADER IMAGE (BASE64) ===============
def get_image_base64(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

header_img_b64 = get_image_base64("why-is-sleep-important.jpg")  # your header image


# =============== GLOBAL STYLES ===============
st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: #f4f5fb;
            font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
        }}

        /* Center page content */
        .main-wrapper {{
            max-width: 1100px;
            margin: 0 auto;
            padding: 1.5rem 0 3rem 0;
        }}

        /* Header / Hero */
        .hero {{
            position: relative;
            border-radius: 18px;
            overflow: hidden;
            margin-bottom: 1.75rem;
            box-shadow: 0 18px 40px rgba(15,23,42,0.32);
        }}
        .hero-bg {{
            background-image: url("data:image/jpg;base64,{header_img_b64}");
            background-size: cover;
            background-position: center;
            width: 100%;
            height: 210px;
            filter: brightness(0.75);
        }}
        .hero-overlay {{
            position: absolute;
            inset: 0;
            background: linear-gradient(90deg, rgba(15,23,42,0.85), rgba(15,23,42,0.35));
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            padding: 1.5rem;
        }}
        .hero-title {{
            font-size: 2.2rem;
            font-weight: 700;
            color: #f9fafb;
            letter-spacing: 0.03em;
        }}
        .hero-subtitle {{
            font-size: 0.95rem;
            color: #e5e7eb;
            max-width: 620px;
            margin-top: 0.4rem;
        }}
        .hero-tag {{
            font-size: 0.8rem;
            color: #e5e7eb;
            padding: 0.25rem 0.8rem;
            border-radius: 999px;
            border: 1px solid rgba(148,163,184,0.7);
            margin-bottom: 0.55rem;
        }}

        /* Cards */
        .card {{
            background: #ffffff;
            border-radius: 14px;
            padding: 1.3rem 1.4rem;
            box-shadow: 0 8px 22px rgba(15,23,42,0.12);
            border: none;
        }}

        .section-title {{
            font-size: 1.05rem;
            font-weight: 600;
            margin-bottom: 0.7rem;
            color: #111827;
        }}

        /* Form labels */
        .stSlider > label,
        .stSelectbox > label,
        .stNumberInput > label {{
            font-size: 0.9rem !important;
            font-weight: 500 !important;
            color: #111827 !important;
        }}

        /* Inputs and selects */
        input, textarea {{
            background-color: #ffffff !important;
            color: #111827 !important;
            border-radius: 8px !important;
        }}
        div[data-baseweb="select"] > div {{
            background-color: #ffffff !important;
            color: #111827 !important;
            border-radius: 8px !important;
        }}
        div[data-testid="stNumberInput"] input {{
            background-color: #ffffff !important;
            color: #111827 !important;
        }}

        /* Result alert text */
        .stAlert p {{
            color: #111827 !important;
            font-weight: 600;
        }}

        /* Metric color */
        div[data-testid="stMetricValue"] {{
            color: #111827 !important;
            font-weight: 700;
        }}

        /* Primary button */
        .stButton > button {{
            width: 100%;
            border-radius: 999px;
            border: none;
            padding: 0.6rem 1rem;
            font-weight: 600;
            background: #2563eb;
            color: #ffffff;
            letter-spacing: 0.02em;
        }}
        .stButton > button:hover {{
            background: #1d4ed8;
            box-shadow: 0 10px 24px rgba(37,99,235,0.35);
        }}

        /* Tabs */
        button[data-baseweb="tab"] {{
            font-weight: 600;
            color: #374151;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# =============== MAIN WRAPPER START ===============
st.markdown('<div class="main-wrapper">', unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown(
    """
    <div class="hero">
        <div class="hero-bg"></div>
        <div class="hero-overlay">
            <div class="hero-tag">Sleep Health · Machine Learning</div>
            <div class="hero-title">Sleep Disorder Prediction Using Machine Learning</div>
            <div class="hero-subtitle">
                An intelligent decision-support tool that predicts whether a patient is more likely
                to have <b>Insomnia</b> or <b>Sleep Apnea</b> using lifestyle and vital-sign features.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =============== TABS ===============
tab_predict, tab_about = st.tabs(["Prediction", "About Model"])

# ---------- PREDICTION TAB ----------
with tab_predict:
    col_left, col_right = st.columns([1.8, 1.1])

    # ---- LEFT: INPUT FORM ----
    with col_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        c1, c2 = st.columns(2)

        with c1:
            st.markdown('<div class="section-title">Patient Profile</div>', unsafe_allow_html=True)
            gender_str = st.selectbox("Gender", list(le_gender.classes_))
            age = st.number_input("Age (years)", min_value=10, max_value=90, value=30)
            occupation_str = st.selectbox("Occupation", list(le_occupation.classes_))
            bmi_str = st.selectbox("BMI Category", list(le_bmi.classes_))

        with c2:
            st.markdown('<div class="section-title">Sleep & Lifestyle</div>', unsafe_allow_html=True)
            sleep_duration = st.number_input(
                "Sleep Duration (hours / night)",
                min_value=0.0,
                max_value=12.0,
                value=6.5,
                step=0.1,
            )
            quality_of_sleep = st.slider("Quality of Sleep (1 – 10)", 1, 10, 7)
            physical_activity = st.slider("Physical Activity (minutes / day)", 0, 300, 60)
            stress_level = st.slider("Stress Level (1 – 10)", 1, 10, 5)

        st.markdown("---")

        c3, c4, c5 = st.columns(3)
        with c3:
            heart_rate = st.number_input("Heart Rate (bpm)", 40, 150, 75)
        with c4:
            daily_steps = st.number_input("Daily Steps", 0, 40000, 8000, step=500)
        with c5:
            systolic_bp = st.number_input("Systolic BP", 80, 220, 130)
            diastolic_bp = st.number_input("Diastolic BP", 40, 140, 85)

        st.markdown("")
        predict_clicked = st.button("🔍 Predict Sleep Disorder")

        st.markdown('</div>', unsafe_allow_html=True)

    # ---- RIGHT: RESULT PANEL ----
    with col_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Result</div>', unsafe_allow_html=True)

        result_box = st.empty()

        if predict_clicked:
            try:
                # Encode categoricals as in training
                gender = le_gender.transform([gender_str])[0]
                occupation = le_occupation.transform([occupation_str])[0]
                bmi_cat = le_bmi.transform([bmi_str])[0]

                feature_values = {
                    "Gender": gender,
                    "Age": age,
                    "Occupation": occupation,
                    "Sleep Duration": sleep_duration,
                    "Quality of Sleep": quality_of_sleep,
                    "Physical Activity Level": physical_activity,
                    "Stress Level": stress_level,
                    "BMI Category": bmi_cat,
                    "Heart Rate": heart_rate,
                    "Daily Steps": daily_steps,
                    "Systolic_BP": systolic_bp,
                    "Diastolic_BP": diastolic_bp,
                }
                row = [feature_values[col] for col in feature_cols]
                sample = np.array([row])

                sample_scaled = scaler.transform(sample)
                pred_encoded = int(model.predict(sample_scaled)[0])
                proba = model.predict_proba(sample_scaled)[0]
                pred_label = le_sleep.inverse_transform([pred_encoded])[0]
                confidence = float(proba[pred_encoded] * 100)

                if pred_label == "Insomnia":
                    result_box.warning(f"Predicted Disorder: **{pred_label}**")
                elif pred_label == "Sleep Apnea":
                    result_box.error(f"Predicted Disorder: **{pred_label}**")
                else:
                    result_box.info(f"Predicted Class: **{pred_label}**")

                st.metric("Model Confidence", f"{confidence:.1f} %")
                st.caption(
                    "Confidence represents the model's estimated probability for the predicted class."
                )

            except Exception as e:
                result_box.error(f"Error while predicting: {e}")
        else:
            result_box.info("Fill the form on the left and click **Predict Sleep Disorder** to see the result.")

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Note</div>', unsafe_allow_html=True)
        st.write(
            "This application is developed as an academic decision-support prototype for a final-year project. "
            "It must not be used as a standalone diagnostic tool. Clinical decisions should always be taken by "
            "qualified healthcare professionals."
        )
        st.markdown('</div>', unsafe_allow_html=True)

# ---------- ABOUT TAB ----------
with tab_about:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Project Overview</div>', unsafe_allow_html=True)
    st.write(
        """
        **Title:** Sleep Disorder Prediction Using Machine Learning  

        **Objective:**  
        To classify patients into *Insomnia* or *Sleep Apnea* categories using demographic,
        lifestyle and physiological parameters.

        **Workflow Summary:**
        - Data cleaning and handling of missing values  
        - Feature engineering (splitting blood pressure into systolic/diastolic components)  
        - Label encoding of categorical variables  
        - Feature scaling using StandardScaler  
        - Model training with Random Forest Classifier  
        - Evaluation using accuracy and classification report  
        - Deployment using Streamlit as an interactive web interface  
        """
    )
    st.markdown("---")
    st.markdown('<div class="section-title">Disclaimer</div>', unsafe_allow_html=True)
    st.write(
        "This tool is part of an academic final-year project and is not a certified medical device. "
        "The outputs are for demonstration and learning purposes only."
    )
    st.markdown('</div>', unsafe_allow_html=True)

# =============== MAIN WRAPPER END ===============
st.markdown('</div>', unsafe_allow_html=True)
