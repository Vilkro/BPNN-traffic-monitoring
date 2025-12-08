import streamlit as st
import numpy as np
import pandas as pd

from framework import AdaptiveMonitoringFramework

BASE_PATH = r"D:/TrafficClassification/MachineLearningCVE"

CSV_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
]

PHASE1_DIR = "saved_models/phase1_mon_thu"       # Mon–Thu only
PHASE2_DIR = "saved_models/phase2_adapted"       # Mon–Thu + Friday


@st.cache_resource
def init_framework():
    """
    Load CICIDS data (once) and prepare Mon–Thu / Friday splits.
    Models and preprocessing will be loaded separately from saved_models.
    """
    fw = AdaptiveMonitoringFramework(BASE_PATH, CSV_FILES)
    fw.load_and_prepare()
    return fw


def load_model_version(framework, version: str):
    """
    version: 'baseline' or 'adapted'
    """
    if version == "baseline":
        framework.load_state(PHASE1_DIR, version_name="mon_thu_only")
    elif version == "adapted":
        framework.load_state(PHASE2_DIR, version_name="mon_thu_plus_fri")
    else:
        st.error("Unknown model version.")
        return

    st.success(f"Loaded model version: {version}")


def evaluate_on_friday_subset(framework, n_samples_per_class=2000):
    """
    Take a balanced subset of Friday data and evaluate current model.
    """
    df_fri = framework.df_fri.copy()
    if "LabelEnc" not in df_fri.columns:
        st.error("Friday subset has no 'LabelEnc' column.")
        return

    # Balanced sampling
    groups = []
    for lab, g in df_fri.groupby("LabelEnc"):
        take = min(len(g), n_samples_per_class)
        groups.append(g.sample(n=take, random_state=42))
    df_bal = pd.concat(groups, ignore_index=True)

    X, y = framework.prep.prepare_features(df_bal, "LabelEnc")
    X_scaled = framework.prep.transform(X)

    metrics = framework.model_manager.evaluate(
        X_scaled, y,
        framework.label_encoder,
        subset_name="[UI] Friday balanced subset"
    )

    st.write("### Evaluation on Friday balanced subset")
    st.write(f"- Accuracy: **{metrics['accuracy']:.4f}**")
    st.write(f"- F1-macro: **{metrics['f1_macro']:.4f}**")
    st.write(f"- F1-weighted: **{metrics['f1_weighted']:.4f}**")

    with st.expander("Show full classification report"):
        st.text(metrics["report"])


def demo_random_flows(framework, n_rows=20):
    """
    Show random Friday flows with predicted and true labels.
    """
    df_fri = framework.df_fri.copy()
    df_sample = df_fri.sample(n=min(n_rows, len(df_fri)), random_state=123).reset_index(drop=True)

    # Prepare features using saved numeric columns
    if framework.prep.num_cols is None:
        st.error("Numeric columns (num_cols) not loaded.")
        return

    num_cols = framework.prep.num_cols
    # Some columns might be missing if training used a slightly different subset
    num_cols_present = [c for c in num_cols if c in df_sample.columns]

    X = df_sample[num_cols_present].copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X = X.fillna(X.median(numeric_only=True)).astype("float32")
    X_scaled = framework.prep.transform(X)

    y_pred_idx = framework.model_manager.predict(X_scaled)
    y_pred_labels = framework.prep.label_encoder.inverse_transform(y_pred_idx)

    true_labels = df_sample["Label"] if "Label" in df_sample.columns else None

    result_df = pd.DataFrame()
    # Show a few basic columns if present
    for col in ["SourceFile", "Timestamp", "Flow ID", "Src IP", "Dst IP"]:
        if col in df_sample.columns:
            result_df[col] = df_sample[col]
    if true_labels is not None:
        result_df["TrueLabel"] = true_labels
    result_df["PredictedLabel"] = y_pred_labels

    st.write("### Random Friday flows with predictions")
    st.dataframe(result_df)


def classify_uploaded_csv(framework):
    """
    Let user upload a CSV with flows and classify them using the current model.
    """
    st.write("### Classify uploaded CSV")

    uploaded_file = st.file_uploader("Upload a CSV file with flow-level features", type=["csv"])
    if uploaded_file is None:
        return

    df = pd.read_csv(uploaded_file)
    st.write("Uploaded data shape:", df.shape)

    # Clean like in initial pipeline
    df_clean = framework.prep.initial_clean(df)

    # Use saved numeric columns from training
    if framework.prep.num_cols is None:
        st.error("Numeric column list (num_cols) not available.")
        return

    num_cols = framework.prep.num_cols
    num_cols_present = [c for c in num_cols if c in df_clean.columns]
    if not num_cols_present:
        st.error("No matching numeric columns found in uploaded CSV.")
        return

    X = df_clean[num_cols_present].copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X = X.fillna(X.median(numeric_only=True)).astype("float32")

    X_scaled = framework.prep.transform(X)
    y_pred_idx = framework.model_manager.predict(X_scaled)
    y_pred_labels = framework.prep.label_encoder.inverse_transform(y_pred_idx)

    df_result = df_clean.copy()
    df_result["PredictedLabel"] = y_pred_labels

    st.write("### Prediction results")
    st.dataframe(df_result.head(100))

    st.download_button(
        label="Download full predictions as CSV",
        data=df_result.to_csv(index=False).encode("utf-8"),
        file_name="classified_flows.csv",
        mime="text/csv"
    )


def main():
    st.set_page_config(page_title="BPNN Telecom Monitoring Framework", layout="wide")

    st.title("📡 BPNN-based Telecom Traffic Monitoring Framework")
    st.write("Lightweight, adaptive & reconfigurable anomaly detection on CICIDS-2017.")

    # Init framework (load data once)
    framework = init_framework()

    # Sidebar: select model version
    st.sidebar.header("Model control")
    model_version = st.sidebar.radio(
        "Choose model version",
        ["baseline (Mon–Thu only)", "adapted (Mon–Thu + Friday)"],
        index=1
    )

    if "current_version" not in st.session_state:
        st.session_state["current_version"] = None

    if model_version.startswith("baseline"):
        version_key = "baseline"
    else:
        version_key = "adapted"

    if st.session_state["current_version"] != version_key:
        # Load chosen model version
        with st.spinner("Loading selected model version..."):
            load_model_version(framework, version_key)
        st.session_state["current_version"] = version_key

    st.sidebar.markdown("---")
    st.sidebar.write(f"**Active model:** `{st.session_state['current_version']}`")

    tab_overview, tab_eval, tab_demo, tab_upload = st.tabs(
        ["Overview", "Evaluate on Friday subset", "Random flows demo", "Classify uploaded CSV"]
    )

    with tab_overview:
        st.write("### Framework overview")
        st.markdown("""
        This interface demonstrates the proposed **BPNN-based monitoring framework**:

        - Uses **Backpropagation Neural Network (BPNN)** as the core classifier.
        - Operates on **flow-level features** from CICIDS-2017 (telecom-style traffic).
        - Supports **adaptation** and **reconfiguration** via:
          - Model version `baseline` (Mon–Thu only).
          - Model version `adapted` (Mon–Thu + Friday traffic).
        - Allows:
          - Evaluation on Friday subset.
          - Visual inspection of predictions on random flows.
          - Classification of user-uploaded traffic CSVs.
        """)

        st.info("""
        Try switching between **baseline** and **adapted** models in the sidebar,
        then compare their behavior on Friday traffic in the other tabs.
        """)

    with tab_eval:
        st.write("Evaluate the current model on a subset of Friday traffic.")
        n_per_class = st.slider(
            "Number of samples per class (Friday subset)",
            min_value=500,
            max_value=5000,
            value=2000,
            step=500
        )
        if st.button("Run evaluation on Friday subset"):
            with st.spinner("Evaluating on Friday subset..."):
                evaluate_on_friday_subset(framework, n_samples_per_class=n_per_class)

    with tab_demo:
        st.write("Inspect random Friday flows and see predicted vs true labels.")
        n_rows = st.slider("Number of random flows to show", 5, 50, 20, 5)
        if st.button("Show random flows"):
            with st.spinner("Sampling and classifying..."):
                demo_random_flows(framework, n_rows=n_rows)

    with tab_upload:
        classify_uploaded_csv(framework)


if __name__ == "__main__":
    main()

