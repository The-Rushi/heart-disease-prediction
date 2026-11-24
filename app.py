from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------------
# Page configuration and visual theme
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Prediction | Framingham Study",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
        --ink: #172b4d;
        --muted: #64748b;
        --blue: #2457c5;
        --teal: #0f9d91;
        --line: #e5eaf2;
        --surface: #ffffff;
        --canvas: #f5f8fc;
    }
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        color: var(--ink);
    }
    .stApp { background: var(--canvas); }
    [data-testid="stHeader"] { background: rgba(245,248,252,0.92); }
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] * { color: var(--ink); }
    h1, h2, h3 {
        font-family: 'Manrope', sans-serif !important;
        color: var(--ink) !important;
        letter-spacing: -0.03em;
    }
    h1 { font-size: 2.15rem !important; font-weight: 800 !important; }
    h2 { font-size: 1.45rem !important; font-weight: 750 !important; }
    h3 { font-size: 1.08rem !important; font-weight: 700 !important; }
    p, label, .stMarkdown { color: #334155; }
    .hero {
        background: linear-gradient(120deg, #ffffff 0%, #edf5ff 58%, #e9f8f6 100%);
        border: 1px solid #e1eaf5;
        border-radius: 22px;
        padding: 28px 30px;
        margin: 4px 0 22px 0;
    }
    .hero-kicker {
        color: #2457c5;
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: .13em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .hero-title {
        font-family: 'Manrope', sans-serif;
        color: #172b4d;
        font-size: 2rem;
        line-height: 1.18;
        font-weight: 800;
        margin: 0 0 10px 0;
    }
    .hero-copy { color: #53657d; font-size: 1rem; max-width: 850px; }
    .metric-card {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 17px 18px;
        min-height: 108px;
        box-shadow: 0 3px 12px rgba(20,45,80,.035);
    }
    .metric-label {
        color: #64748b;
        font-size: .78rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .metric-value {
        color: #172b4d;
        font-family: 'Manrope', sans-serif;
        font-size: 1.75rem;
        font-weight: 800;
        line-height: 1;
    }
    .metric-note { color: #8290a4; font-size: .74rem; margin-top: 7px; }
    .section-card {
        background: white;
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 20px 22px;
        margin-bottom: 16px;
    }
    .result-positive {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        border-left: 5px solid #ea580c;
        border-radius: 16px;
        padding: 20px;
    }
    .result-negative {
        background: #effaf6;
        border: 1px solid #b8e8d7;
        border-left: 5px solid #0f9d75;
        border-radius: 16px;
        padding: 20px;
    }
    .small-muted { color: #64748b; font-size: .85rem; }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #2457c5, #147f9b);
        border: 0;
        border-radius: 11px;
        color: white;
        font-weight: 700;
        min-height: 3rem;
    }
    div.stButton > button[kind="primary"]:hover {
        border: 0; color: white; filter: brightness(1.06);
    }
    [data-testid="stForm"] {
        background: white;
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 20px;
    }
    .disclaimer {
        background: #fffaf0;
        border: 1px solid #f5dfad;
        border-radius: 12px;
        padding: 13px 16px;
        color: #77551c;
        font-size: .83rem;
    }
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"


# ---------------------------------------------------------------------
# Load exported artifacts
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the trained prediction pipeline...")
def load_model(model_path: str):
    return joblib.load(model_path)


@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)


def artifact_status() -> tuple[bool, list[str]]:
    required = [
        MODEL_DIR / "best_model.joblib",
        MODEL_DIR / "model_metadata.json",
        MODEL_DIR / "model_comparison_validation.csv",
        MODEL_DIR / "final_test_metrics.csv",
    ]
    missing = [str(path.relative_to(BASE_DIR)) for path in required if not path.exists()]
    return not missing, missing


available, missing_files = artifact_status()
if not available:
    st.error("Some model files are missing. Place the exported files inside the `models/` folder.")
    st.code("\n".join(missing_files))
    st.info(
        "Download and extract the complete `models` folder generated by the notebook. "
        "Keep the folder beside `app.py`."
    )
    st.stop()

try:
    metadata = load_json(str(MODEL_DIR / "model_metadata.json"))
    model = load_model(str(MODEL_DIR / "best_model.joblib"))
    validation_df = load_csv(str(MODEL_DIR / "model_comparison_validation.csv"))
    test_df = load_csv(str(MODEL_DIR / "final_test_metrics.csv"))
except Exception as exc:
    st.error(f"Could not load the exported model artifacts: {exc}")
    st.stop()

features = metadata.get(
    "features",
    [
        "male", "age", "education", "currentSmoker", "cigsPerDay",
        "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes",
        "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose",
    ],
)
age_min = int(metadata.get("age_min_observed", 32))
age_max = int(metadata.get("age_max_observed", 70))
selected_model_name = metadata.get("selected_model", "Selected model")
threshold = float(metadata.get("decision_threshold", 0.5))


# ---------------------------------------------------------------------
# Shared components
# ---------------------------------------------------------------------
def hero(title: str, subtitle: str, kicker: str = "FRAMINGHAM HEART STUDY"):
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-kicker">{kicker}</div>
          <div class="hero-title">{title}</div>
          <div class="hero-copy">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, note: str = ""):
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def safe_float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be a valid number.")
    if not pd.notna(result):
        raise ValueError(f"{field} is required.")
    return result


def make_prediction(values: dict[str, Any]) -> tuple[int, float]:
    row = pd.DataFrame([{feature: values[feature] for feature in features}], columns=features)
    probability = float(model.predict_proba(row)[0, 1])
    prediction = int(probability >= threshold)
    return prediction, probability


# ---------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="padding:8px 4px 18px 4px">
          <div style="font-size:1.55rem;font-weight:800;color:#172b4d;font-family:Manrope,sans-serif">
            ♥ Cardio<span style="color:#0f9d91">Predict</span>
          </div>
          <div style="font-size:.78rem;color:#8290a4">Machine learning project</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    page = st.radio(
        "NAVIGATION",
        ["Overview", "Prediction", "Model comparison", "About"],
        label_visibility="visible",
    )
    st.divider()
    st.caption("MODEL INFORMATION")
    st.markdown(f"**Selected model:** {selected_model_name}")
    st.markdown(f"**Decision threshold:** `{threshold:.3f}`")
    st.markdown(f"**Training data age range:** {age_min}–{age_max} years")
    st.divider()
    st.caption("Academic demonstration · Not for clinical use")


# ---------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------
if page == "Overview":
    hero(
        "Understand the model. Explore the outcome.",
        "An academic machine learning demonstration that uses the Framingham Heart Study dataset "
        "to classify the observed 10-year coronary heart disease outcome.",
    )
    st.markdown("### Project at a glance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset records", "4,240")
    c2.metric("Input features", "15")
    c3.metric("Models compared", "5")
    c4.metric("Selected model", selected_model_name)

    st.markdown("")
    left, right = st.columns([1.25, 1])
    with left:
        st.markdown(
            """
            <div class="section-card">
              <h3>What does this application do?</h3>
              <p>It accepts 15 input variables from the Framingham dataset and applies a previously
              trained machine learning pipeline to predict the dataset's binary <b>TenYearCHD</b> outcome.</p>
              <ul>
                <li>Uses a saved preprocessing and model pipeline.</li>
                <li>Applies the decision threshold selected on validation data.</li>
                <li>Shows model evaluation results from the notebook.</li>
                <li>Does not train or modify the model when the app starts.</li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            """
            <div class="section-card">
              <h3>How to use</h3>
              <ol>
                <li>Open the <b>Prediction</b> page.</li>
                <li>Enter the 15 required dataset features.</li>
                <li>Submit the form.</li>
                <li>Review the predicted class and model score.</li>
              </ol>
              <p class="small-muted">Inputs should reflect the dataset's feature definitions and units.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="disclaimer"><b>Important:</b> This is an academic demonstration. '
        'It is not a medical device, diagnosis, or clinically validated individual risk calculator. '
        'Do not use its output to make healthcare decisions.</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------
# Prediction page
# ---------------------------------------------------------------------
elif page == "Prediction":
    hero(
        "10-year CHD outcome prediction",
        "Enter the participant attributes using the same definitions and units as the Framingham dataset.",
        "PREDICTION WORKSPACE",
    )
    st.info(
        f"The training dataset contains participants aged {age_min}–{age_max}. "
        "This application rejects ages outside that observed range."
    )

    with st.form("prediction_form"):
        st.markdown("### 1. Personal information")
        a1, a2, a3 = st.columns(3)
        with a1:
            age = st.number_input("Age (years)", min_value=1, max_value=120, value=50, step=1)
        with a2:
            sex_label = st.selectbox("Sex (dataset coding)", ["Female (0)", "Male (1)"])
            male = 1 if sex_label.startswith("Male") else 0
        with a3:
            education_label = st.selectbox(
                "Education level (dataset code)",
                ["1", "2", "3", "4"],
                index=1,
                help="Enter the original Framingham education category code; this is not a free-text qualification.",
            )
            education = int(education_label)

        st.markdown("### 2. Smoking and medical history")
        b1, b2, b3 = st.columns(3)
        with b1:
            current_smoker = st.selectbox("Currently smokes?", ["No", "Yes"])
            currentSmoker = int(current_smoker == "Yes")
            cigs_per_day = st.number_input(
                "Cigarettes per day", min_value=0, max_value=100, value=0, step=1
            )
        with b2:
            bp_meds_label = st.selectbox("Taking blood pressure medication?", ["No", "Yes"])
            BPMeds = int(bp_meds_label == "Yes")
            stroke_label = st.selectbox("Previous stroke?", ["No", "Yes"])
            prevalentStroke = int(stroke_label == "Yes")
        with b3:
            hypertension_label = st.selectbox("Prevalent hypertension?", ["No", "Yes"])
            prevalentHyp = int(hypertension_label == "Yes")
            diabetes_label = st.selectbox("Diabetes?", ["No", "Yes"])
            diabetes = int(diabetes_label == "Yes")

        st.markdown("### 3. Health measurements")
        d1, d2, d3 = st.columns(3)
        with d1:
            totChol = st.number_input("Total cholesterol (mg/dL)", min_value=50.0, max_value=700.0, value=200.0, step=1.0)
            sysBP = st.number_input("Systolic blood pressure (mmHg)", min_value=50.0, max_value=300.0, value=120.0, step=1.0)
        with d2:
            diaBP = st.number_input("Diastolic blood pressure (mmHg)", min_value=30.0, max_value=200.0, value=80.0, step=1.0)
            bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=80.0, value=25.0, step=0.1)
        with d3:
            heartRate = st.number_input("Heart rate (beats/min)", min_value=30.0, max_value=220.0, value=70.0, step=1.0)
            glucose = st.number_input("Glucose (mg/dL)", min_value=20.0, max_value=600.0, value=80.0, step=1.0)

        submitted = st.form_submit_button("Run prediction", type="primary", use_container_width=True)

    if submitted:
        values = {
            "male": male,
            "age": int(age),
            "education": education,
            "currentSmoker": currentSmoker,
            "cigsPerDay": int(cigs_per_day),
            "BPMeds": BPMeds,
            "prevalentStroke": prevalentStroke,
            "prevalentHyp": prevalentHyp,
            "diabetes": diabetes,
            "totChol": float(totChol),
            "sysBP": float(sysBP),
            "diaBP": float(diaBP),
            "BMI": float(bmi),
            "heartRate": float(heartRate),
            "glucose": float(glucose),
        }

        if not age_min <= int(age) <= age_max:
            st.error(f"Age must be between {age_min} and {age_max} years, matching the observed dataset range.")
        elif cigs_per_day == 0 and currentSmoker == 1:
            st.warning("You selected current smoker but entered zero cigarettes per day. Please verify these inputs.")
        elif cigs_per_day > 0 and currentSmoker == 0:
            st.warning("You entered cigarettes per day but selected not currently smoking. Please verify these inputs.")
        else:
            try:
                predicted_class, score = make_prediction(values)
                st.markdown("### Prediction result")
                result_class = "result-positive" if predicted_class == 1 else "result-negative"
                result_title = (
                    "Positive CHD outcome class predicted"
                    if predicted_class == 1
                    else "Negative CHD outcome class predicted"
                )
                result_description = (
                    "The model classifies this input into the positive TenYearCHD class."
                    if predicted_class == 1
                    else "The model classifies this input into the negative TenYearCHD class."
                )
                st.markdown(
                    f'<div class="{result_class}"><h3>{result_title}</h3>'
                    f'<p>{result_description}</p></div>',
                    unsafe_allow_html=True,
                )
                st.write("")
                p1, p2, p3 = st.columns(3)
                p1.metric("Model score", f"{score:.1%}")
                p2.metric("Decision threshold", f"{threshold:.1%}")
                p3.metric("Model", selected_model_name)
                st.caption(
                    "Model score is the classifier's output for the positive class. "
                    "It is not a calibrated medical risk percentage."
                )
                with st.expander("View submitted feature values"):
                    st.dataframe(
                        pd.DataFrame([values]).T.rename(columns={0: "Input value"}),
                        use_container_width=True,
                    )
            except Exception as exc:
                st.error(f"Prediction could not be completed: {exc}")

    st.markdown(
        '<div class="disclaimer"><b>Medical disclaimer:</b> This prediction is for academic '
        'demonstration only. It is not a diagnosis, medical advice, or a clinically validated '
        'risk estimate. Consult a qualified healthcare professional for medical concerns.</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------
# Model comparison page
# ---------------------------------------------------------------------
elif page == "Model comparison":
    hero(
        "Model performance",
        "Compare the five classifiers using the validation results generated by the training notebook.",
        "EVALUATION DASHBOARD",
    )

    display_df = validation_df.copy()
    display_df.index.name = "Model"
    preferred = ["Recall", "Specificity", "Precision", "F1", "ROC_AUC", "PR_AUC", "Accuracy", "FNR"]
    present = [col for col in preferred if col in display_df.columns]

    st.markdown("### Validation comparison")
    st.caption("These are validation metrics, not final test metrics.")
    metric_cols = st.columns(4)
    top_row = display_df.sort_values("Recall", ascending=False).iloc[0]
    metric_cols[0].metric("Highest validation recall", f"{top_row['Recall']:.1%}")
    metric_cols[1].metric("Best recall model", str(display_df["Recall"].idxmax()))
    metric_cols[2].metric("Selected model", selected_model_name)
    metric_cols[3].metric("Selected threshold", f"{threshold:.3f}")

    st.dataframe(
        display_df[present].style.format("{:.3f}"),
        use_container_width=True,
    )

    if "Recall" in display_df.columns:
        st.markdown("### Recall by model")
        recall_plot = display_df.reset_index()[["Model", "Recall"]].sort_values("Recall", ascending=True)
        fig = px.bar(
            recall_plot,
            x="Recall",
            y="Model",
            orientation="h",
            text=recall_plot["Recall"].map(lambda x: f"{x:.1%}"),
            color="Recall",
            color_continuous_scale=["#b9d7f5", "#2457c5"],
        )
        fig.update_layout(
            height=360, plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(range=[0, 1], tickformat=".0%", title="Recall"),
            yaxis_title="", coloraxis_showscale=False,
            margin=dict(l=10, r=20, t=15, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    chart_metrics = [m for m in ["Recall", "Specificity", "Precision", "ROC_AUC", "PR_AUC"] if m in display_df.columns]
    if chart_metrics:
        st.markdown("### Metric comparison")
        long_df = display_df[chart_metrics].reset_index().melt(
            id_vars="Model", var_name="Metric", value_name="Score"
        )
        fig2 = px.bar(
            long_df, x="Model", y="Score", color="Metric", barmode="group",
            color_discrete_sequence=["#2457c5", "#0f9d91", "#70a7df", "#8b9bb4", "#63b7ae"],
        )
        fig2.update_layout(
            height=420, plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(range=[0, 1], title="Score"),
            xaxis_title="", legend_title="Metric",
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Selected model — final test evaluation")
    if not test_df.empty:
        test_metrics = test_df.iloc[0]
        test_cols = st.columns(4)
        for col, metric, label in zip(
            test_cols,
            ["Recall", "Specificity", "Precision", "ROC_AUC"],
            ["Test recall", "Test specificity", "Test precision", "Test ROC-AUC"],
        ):
            if metric in test_metrics:
                value = float(test_metrics[metric])
                col.metric(label, f"{value:.3f}" if metric == "ROC_AUC" else f"{value:.1%}")
        if all(k in test_metrics for k in ["TN", "FP", "FN", "TP"]):
            cm_df = pd.DataFrame(
                [[int(test_metrics["TN"]), int(test_metrics["FP"])],
                 [int(test_metrics["FN"]), int(test_metrics["TP"])]],
                index=["Actual negative", "Actual positive"],
                columns=["Predicted negative", "Predicted positive"],
            )
            st.markdown("#### Test confusion matrix")
            st.dataframe(cm_df, use_container_width=True)

    st.markdown(
        '<div class="disclaimer"><b>Interpretation:</b> Recall, specificity and precision describe '
        'different error trade-offs. A higher recall alone does not establish that a model is suitable '
        'for clinical use.</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------
# About page
# ---------------------------------------------------------------------
elif page == "About":
    hero(
        "About this project",
        "A student machine learning project exploring binary classification of the 10-year CHD outcome.",
        "PROJECT INFORMATION",
    )
    st.markdown(
        """
        <div class="section-card">
          <h3>Dataset and target</h3>
          <p>The application is based on the Framingham Heart Study dataset distributed as
          <code>framingham.csv</code>. The target variable is <code>TenYearCHD</code>:
          0 represents no recorded CHD event during the follow-up and 1 represents a recorded
          CHD event during the follow-up.</p>
          <p>The target is an observed dataset outcome. It should not be interpreted as a
          clinical diagnosis or a guaranteed individual risk estimate.</p>
        </div>
        <div class="section-card">
          <h3>Models evaluated</h3>
          <ul>
            <li>Logistic Regression</li>
            <li>Decision Tree</li>
            <li>Random Forest</li>
            <li>Support Vector Machine (SVM)</li>
            <li>XGBoost</li>
          </ul>
        </div>
        <div class="section-card">
          <h3>Methodology</h3>
          <p>The notebook separates training, validation and test data. Preprocessing is included
          in the fitted model pipeline. Hyperparameter tuning is performed using training folds,
          threshold selection uses validation data, and final performance is reported on the
          held-out test set.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### Model metadata")
    metadata_view = {
        "Selected model": selected_model_name,
        "Decision threshold": threshold,
        "Age range in dataset": f"{age_min}–{age_max}",
        "Target": metadata.get("target", "TenYearCHD"),
        "Model selection rule": metadata.get("model_selection_rule", "Not provided"),
        "Threshold selection rule": metadata.get("threshold_selection_rule", "Not provided"),
    }
    st.json(metadata_view)

    st.markdown(
        '<div class="disclaimer"><b>Academic and medical disclaimer:</b> This application is '
        'created for educational demonstration. It has not been clinically validated, is not a '
        'medical device, and must not be used for diagnosis, treatment, or healthcare decisions.</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------
st.markdown(
    """
    <div style="text-align:center;color:#94a3b8;font-size:.76rem;padding:26px 0 10px 0">
      HEART DISEASE PREDICTION · FRAMINGHAM DATASET · ACADEMIC MACHINE LEARNING PROJECT
    </div>
    """,
    unsafe_allow_html=True,
)
