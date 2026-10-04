import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from contextlib import contextmanager

import joblib
import numpy as np
import pandas as pd
import streamlit as st

PROCESSED_DIR = "data/processed"
MODEL_DIR = "outputs/models"

LABEL_COLUMNS_TO_STRIP = ["Label", "label", "label_binary", "label_multiclass"]


@contextmanager
def timer(label):
    start = time.time()
    yield
    st.session_state.setdefault("timings", []).append((label, time.time() - start))


# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Cyber Anomaly Detector",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Custom Styling
# ---------------------------------------------------------
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">

<style>
    /* Global Font & Canvas Background */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #F6F5FB !important;
        color: #141226 !important;
    }

    /* Container Spacing */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 4rem !important;
        max-width: 1100px !important;
    }

    /* Top Floating Pill Navbar */
    .bloom-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        padding: 0.9rem 1.8rem;
        border-radius: 9999px;
        border: 1px solid rgba(220, 215, 240, 0.8);
        margin-bottom: 2.2rem;
        box-shadow: 0 4px 20px rgba(80, 60, 150, 0.04);
    }
    .bloom-brand {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        font-weight: 700;
        font-size: 1.05rem;
        color: #141226;
        letter-spacing: -0.01em;
    }
    .bloom-brand span {
        color: #7B61FF;
        font-size: 1.25rem;
    }
    .bloom-nav-badge {
        background: #17152A;
        color: #FFFFFF !important;
        padding: 0.45rem 1.2rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    /* Hero Section (Clean & Minimal) */
    .hero-container {
        text-align: center;
        padding: 2rem 1.5rem 3rem 1.5rem;
    }
    .hero-title {
        font-size: 3.2rem;
        font-weight: 800;
        color: #151329;
        letter-spacing: -0.035em;
        line-height: 1.15;
        margin: 0 auto;
        max-width: 800px;
    }

    /* Bento Feature Cards */
    .bento-grid {
        display: grid;
        grid-template-columns: 1.3fr 1fr 1fr;
        gap: 1.2rem;
        margin-bottom: 2.8rem;
    }
    @media (max-width: 820px) {
        .bento-grid { grid-template-columns: 1fr; }
    }
    .bento-card-light {
        background: linear-gradient(135deg, #E6E1FA 0%, #D8CEF8 100%);
        border-radius: 24px;
        padding: 2rem;
        color: #1A1733;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 220px;
        box-shadow: 0 10px 30px rgba(110, 90, 200, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.6);
    }
    .bento-card-dark {
        background: #17152B;
        border-radius: 24px;
        padding: 2rem;
        color: #FFFFFF;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 220px;
        box-shadow: 0 10px 30px rgba(20, 15, 45, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    .bento-title {
        font-size: 1.35rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.6rem;
    }
    .bento-desc {
        font-size: 0.9rem;
        line-height: 1.55;
    }
    .bento-card-light .bento-desc { color: #433E60; }
    .bento-card-dark .bento-desc { color: #A4A1BD; }

    /* Custom Streamlit Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #EBE8F5 !important;
        border-radius: 9999px !important;
        padding: 5px !important;
        gap: 6px !important;
        border-bottom: none !important;
        display: inline-flex !important;
        margin-bottom: 1.8rem !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 9999px !important;
        padding: 0.55rem 1.6rem !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        color: #65627A !important;
        border: none !important;
        background: transparent !important;
        transition: all 0.2s ease !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #17152B !important;
        box-shadow: 0 4px 14px rgba(25, 20, 50, 0.08) !important;
    }
    .stTabs [data-baseweb="tab-border"] { display: none !important; }

    /* Metric Cards */
    [data-testid="stMetric"] {
        background: #FFFFFF !important;
        border: 1px solid #E8E5F3 !important;
        border-radius: 20px !important;
        padding: 1.4rem !important;
        box-shadow: 0 4px 20px rgba(50, 40, 90, 0.03) !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        color: #726F88 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.03em !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 800 !important;
        color: #16142A !important;
        letter-spacing: -0.03em !important;
    }

    /* Button Styling */
    .stButton button {
        border-radius: 9999px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        border: 1px solid transparent !important;
    }
    .stButton button[kind="primary"] {
        background-color: #17152A !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 14px rgba(23, 21, 42, 0.2) !important;
    }
    .stButton button[kind="primary"]:hover {
        background-color: #2D294D !important;
        transform: translateY(-1px) !important;
    }
    .stButton button[kind="secondary"] {
        background-color: #FFFFFF !important;
        color: #2D294D !important;
        border: 1px solid #DEDAF0 !important;
    }
    .stButton button[kind="secondary"]:hover {
        background-color: #F3F1FA !important;
    }

    /* Download Button */
    .stDownloadButton button {
        background-color: #17152A !important;
        color: #FFFFFF !important;
        border-radius: 9999px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        border: none !important;
        box-shadow: 0 4px 14px rgba(23, 21, 42, 0.15) !important;
    }

    /* Status Badges */
    .doc-status {
        padding: 0.9rem 1.1rem;
        border-radius: 16px;
        background-color: #ECFDF5;
        border: 1px solid #A7F3D0;
        color: #065F46;
        font-size: 0.88rem;
        margin-top: 0.8rem;
    }
    .attack-status {
        padding: 0.9rem 1.1rem;
        border-radius: 16px;
        background-color: #FEF2F2;
        border: 1px solid #FECACA;
        color: #991B1B;
        font-size: 0.88rem;
        margin-top: 0.8rem;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #EBE7F5 !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem !important;
    }

    /* Dataframe wrapper */
    [data-testid="stDataFrame"] {
        border-radius: 16px !important;
        overflow: hidden !important;
        border: 1px solid #E6E2F3 !important;
        background: #FFFFFF !important;
        box-shadow: 0 4px 16px rgba(40, 30, 80, 0.02) !important;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Pipeline Loading & Helpers
# ---------------------------------------------------------
@st.cache_resource
def load_pipeline_artifacts():
    try:
        scaler = joblib.load(f"{PROCESSED_DIR}/scaler.joblib")
        feature_columns = joblib.load(f"{PROCESSED_DIR}/feature_columns.joblib")
        rf_binary = joblib.load(f"{MODEL_DIR}/random_forest.joblib")
        xgb_binary = joblib.load(f"{MODEL_DIR}/xgboost.joblib")

        rf_multi, encoder = None, None
        try:
            rf_multi = joblib.load(f"{MODEL_DIR}/random_forest_multiclass.joblib")
            encoder = joblib.load(f"{MODEL_DIR}/label_encoder_multiclass.joblib")
        except FileNotFoundError:
            pass

        return scaler, feature_columns, rf_binary, xgb_binary, rf_multi, encoder
    except FileNotFoundError:
        st.warning(f"Pipeline artifacts not found at '{PROCESSED_DIR}' or '{MODEL_DIR}'. Make sure models are trained.")
        return None, None, None, None, None, None


@st.cache_data
def load_test_set():
    try:
        X_test = pd.read_csv(f"{PROCESSED_DIR}/X_test.csv")
        y_test_binary = pd.read_csv(f"{PROCESSED_DIR}/y_test_binary.csv").squeeze()
        y_test_multi = pd.read_csv(f"{PROCESSED_DIR}/y_test_multiclass.csv").squeeze()
        return X_test, y_test_binary, y_test_multi
    except FileNotFoundError:
        return None, None, None


def strip_label_columns(df: pd.DataFrame):
    cols_found = [c for c in df.columns if c in LABEL_COLUMNS_TO_STRIP]
    if cols_found:
        df = df.drop(columns=cols_found)
    return df, cols_found


def classify_rows(X_scaled, rf_binary, xgb_binary, rf_multi, encoder):
    rf_preds = rf_binary.predict(X_scaled)
    xgb_preds = xgb_binary.predict(X_scaled)

    results = pd.DataFrame({
        "Random Forest": pd.Series(rf_preds).map({0: "Benign", 1: "Attack"}).values,
        "XGBoost (reference)": pd.Series(xgb_preds).map({0: "Benign", 1: "Attack"}).values,
    })

    if rf_multi is not None and encoder is not None:
        attack_mask = rf_preds == 1
        results["Predicted Attack Type"] = "N/A (Benign)"
        if attack_mask.sum() > 0:
            type_preds_enc = rf_multi.predict(X_scaled[attack_mask])
            type_preds = encoder.inverse_transform(type_preds_enc)
            results.loc[attack_mask, "Predicted Attack Type"] = type_preds

    return results


def summarize_results(result: pd.DataFrame):
    total = len(result)
    n_attack = (result["Random Forest"] == "Attack").sum()
    n_normal = total - n_attack

    breakdown = None
    if "Predicted Attack Type" in result.columns:
        breakdown = (
            result.loc[result["Predicted Attack Type"] != "N/A (Benign)",
                       "Predicted Attack Type"]
            .value_counts()
        )

    return total, n_normal, n_attack, breakdown


# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_source" not in st.session_state:
    st.session_state.last_source = None
if "rows_classified" not in st.session_state:
    st.session_state.rows_classified = 0

scaler, feature_columns, rf_binary, xgb_binary, rf_multi, encoder = load_pipeline_artifacts()


# ---------------------------------------------------------
# Sidebar (Left Side: Cyber Anomaly Detector)
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.2rem;">
            <span style="color: #7B61FF; font-size: 1.3rem;">✦</span>
            <span style="font-weight: 800; font-size: 1.15rem; color: #16142A;">Cyber Anomaly Detector</span>
        </div>
        <p style="font-size: 0.8rem; color: #6D6A82; margin-bottom: 1.2rem; line-height: 1.4;">
            Unsupervised clustering (K-Means + Isolation Forest) generates training labels. 
            Random Forest & XGBoost classify real-time incoming traffic.
        </p>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("<p style='font-size: 0.85rem; font-weight: 600; color: #322E4D; margin-bottom: 0.4rem;'>Upload Network Telemetry</p>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload new traffic (CSV)", type="csv", label_visibility="collapsed")

    load_col, clear_col = st.columns([2, 1])
    with load_col:
        load_clicked = st.button("Classify File", use_container_width=True,
                                 type="primary", disabled=uploaded_file is None or scaler is None)
    with clear_col:
        if st.button("Reset", use_container_width=True, type="secondary"):
            st.session_state.last_result = None
            st.session_state.last_source = None
            st.session_state.rows_classified = 0
            st.rerun()

    if uploaded_file and load_clicked and scaler is not None:
        progress = st.progress(0, text="Reading CSV...")
        with timer("Read CSV"):
            raw_df = pd.read_csv(uploaded_file, low_memory=False)
            raw_df.columns = [c.strip() for c in raw_df.columns]

        progress.progress(20, text="Checking for label columns...")
        raw_df, stripped_cols = strip_label_columns(raw_df)
        if stripped_cols:
            st.toast(f"Removed label column(s): {stripped_cols} (preventing leakage)")

        progress.progress(35, text="Cleaning data...")
        with timer("Clean"):
            raw_df = raw_df.replace([np.inf, -np.inf], np.nan).dropna()

        missing = set(feature_columns) - set(raw_df.columns)
        if missing:
            progress.empty()
            st.error(f"Missing required features: {missing}")
        else:
            progress.progress(55, text="Scaling features...")
            with timer("Scale"):
                X_new = raw_df[feature_columns]
                X_scaled = pd.DataFrame(
                    scaler.transform(X_new), columns=X_new.columns, index=X_new.index
                )

            progress.progress(80, text="Running ensemble inference...")
            with timer("Classify"):
                result = classify_rows(X_scaled, rf_binary, xgb_binary, rf_multi, encoder)

            st.session_state.last_result = result
            st.session_state.last_source = uploaded_file.name
            st.session_state.rows_classified = len(result)

            progress.progress(100, text="Inference Complete")
            st.rerun()

    # Status Pill Box
    if st.session_state.last_source and st.session_state.last_result is not None:
        n_attacks = (st.session_state.last_result["Random Forest"] == "Attack").sum()
        box_class = "attack-status" if n_attacks > 0 else "doc-status"
        status_icon = "⚠️" if n_attacks > 0 else "✓"
        st.markdown(
            f'<div class="{box_class}">'
            f'<b>{status_icon} {st.session_state.last_source}</b><br>'
            f'{st.session_state.rows_classified:,} flows inspected · '
            f'<b>{n_attacks:,}</b> flagged as threat'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown("""
            <div style="background: #F3F1FA; border: 1px dashed #D6D0EA; padding: 0.8rem; border-radius: 14px; text-align: center; color: #7B7894; font-size: 0.8rem;">
                No telemetry file classified yet.
            </div>
        """, unsafe_allow_html=True)

    if st.session_state.get("timings"):
        with st.expander("⚡ Latency & Timing Breakdown"):
            for label, secs in st.session_state.timings[-8:]:
                st.write(f"**{label}**: `{secs:.3f}s`")

    st.divider()
    st.caption("You can also run instant evaluations using the held-out test split.")


# ---------------------------------------------------------
# Top Pill Navbar (Network Threat Intelligence & Engine Active)
# ---------------------------------------------------------
st.markdown("""
<div class="bloom-nav">
    <div class="bloom-brand">
        <span>✦</span> Network Threat Intelligence
    </div>
    <div class="bloom-nav-badge">
        Engine Active
    </div>
</div>

<div class="hero-container">
    <div class="hero-title">
        Cyber Anomaly Detector
    </div>
</div>

<div class="bento-grid">
    <div class="bento-card-light">
        <div>
            <div class="bento-title">Intelligence that grows</div>
            <div class="bento-desc">
                Continuous traffic vectorization with auto-sanitization of label leakage and microsecond feature normalization.
            </div>
        </div>
        <div style="font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #6A56C4;">
            ✓ Active Scaler Loaded
        </div>
    </div>
    <div class="bento-card-dark">
        <div>
            <div class="bento-title">Dual-Engine Consensus</div>
            <div class="bento-desc">
                Primary classification backed by Random Forest with concurrent XGBoost validation for high-confidence alerting.
            </div>
        </div>
        <div style="font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #A692FF;">
            RF + XGB Parallel
        </div>
    </div>
    <div class="bento-card-dark">
        <div>
            <div class="bento-title">Zero-Latency Audit</div>
            <div class="bento-desc">
                Granular multi-class decomposition isolating DoS, PortScan, Botnets, and Brute-Force anomalies instantly.
            </div>
        </div>
        <div style="font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #A692FF;">
            Multi-Class Ready
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Interactive Workspace Tabs
# ---------------------------------------------------------
tab1, tab2 = st.tabs(["✦ Try a Real Example", "📊 Uploaded File Results"])

with tab1:
    st.markdown("<h3 style='font-size: 1.35rem; font-weight: 700; color: #17152B; margin-bottom: 0.3rem;'>Real Held-Out Evaluation</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color: #6C6885; font-size: 0.9rem; margin-bottom: 1.2rem;'>Sample a random flow directly from the verified test set to inspect model consensus against the ground truth.</p>", unsafe_allow_html=True)

    X_test, y_test_binary, y_test_multi = load_test_set()

    if X_test is not None and rf_binary is not None:
        btn_col, _ = st.columns([1, 3])
        with btn_col:
            run_sample = st.button("Sample Random Flow", type="primary", use_container_width=True)

        if run_sample:
            idx = X_test.sample(1).index[0]
            row = X_test.loc[[idx]]
            true_binary = "Attack" if y_test_binary.loc[idx] == 1 else "Benign"
            true_multi = y_test_multi.loc[idx]

            with timer("Random example classify"):
                result = classify_rows(row, rf_binary, xgb_binary, rf_multi, encoder)

            rf_pred = result["Random Forest"].iloc[0]
            status_color = "#065F46" if true_binary == "Benign" else "#991B1B"
            bg_color = "#ECFDF5" if true_binary == "Benign" else "#FEF2F2"

            st.markdown(f"""
                <div style="background: {bg_color}; border: 1px solid {status_color}33; border-radius: 16px; padding: 1rem 1.4rem; margin: 1.2rem 0; display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; color: {status_color};">Ground Truth</span>
                        <div style="font-size: 1.25rem; font-weight: 800; color: {status_color};">{true_binary} <span style="font-size: 0.9rem; font-weight: 500; opacity: 0.85;">({true_multi})</span></div>
                    </div>
                    <div style="text-align: right;">
                        <span style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; color: #55526C;">Model Consensus</span>
                        <div style="font-size: 1.25rem; font-weight: 800; color: #17152B;">{rf_pred}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("<p style='font-size: 0.88rem; font-weight: 600; color: #44405E;'>Prediction Output:</p>", unsafe_allow_html=True)
            st.dataframe(result, use_container_width=True)
    else:
        st.info("Held-out test dataset not found in `data/processed`. Train the pipeline to enable live sampling.")

with tab2:
    st.markdown("<h3 style='font-size: 1.35rem; font-weight: 700; color: #17152B; margin-bottom: 0.3rem;'>Batch Traffic Classification</h3>", unsafe_allow_html=True)

    if st.session_state.last_result is not None:
        result = st.session_state.last_result
        total, n_normal, n_attack, breakdown = summarize_results(result)

        st.markdown(f"""
            <div style="margin-bottom: 1.2rem; font-size: 0.92rem; color: #5B5773;">
                Source File: <b style="color: #17152B;">{st.session_state.last_source}</b>
            </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Flows Analyzed", f"{total:,}")
        col2.metric("Benign Normal", f"{n_normal:,}")
        col3.metric("Flagged Threats", f"{n_attack:,}")

        st.markdown("""
            <p style="font-size: 0.82rem; color: #7B7795; margin-top: 0.6rem; margin-bottom: 1.8rem;">
                *Predictions are generated by the trained ensemble models. Flagged items indicate anomalous distribution shifts recommended for automated firewall quarantine.
            </p>
        """, unsafe_allow_html=True)

        if breakdown is not None and len(breakdown) > 0:
            st.markdown("<p style='font-size: 0.95rem; font-weight: 700; color: #1B1832; margin-bottom: 0.4rem;'>Threat Classification Breakdown</p>", unsafe_allow_html=True)
            chart_col, table_col = st.columns([1.5, 1])
            with chart_col:
                st.bar_chart(breakdown, color="#7B61FF")
            with table_col:
                st.dataframe(breakdown.rename("Incidents"), use_container_width=True)

        st.markdown("<p style='font-size: 0.95rem; font-weight: 700; color: #1B1832; margin-top: 1rem; margin-bottom: 0.4rem;'>Detailed Per-Row Inference</p>", unsafe_allow_html=True)
        st.dataframe(result, use_container_width=True)

        st.markdown("<div style='margin-top: 1.2rem;'>", unsafe_allow_html=True)
        st.download_button(
            "Download Flagged CSV Report",
            result.to_csv(index=False),
            file_name="anomaly_predictions.csv",
            mime="text/csv",
        )
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style="background: #FFFFFF; border: 1px dashed #D6D1EA; border-radius: 20px; padding: 3rem 2rem; text-align: center; margin-top: 1rem;">
                <div style="font-size: 2.2rem; margin-bottom: 0.6rem;">📂</div>
                <div style="font-weight: 700; font-size: 1.1rem; color: #17152B; margin-bottom: 0.3rem;">No Telemetry Data Uploaded</div>
                <p style="color: #726E8B; font-size: 0.88rem; max-width: 440px; margin: 0 auto;">
                    Select a CSV traffic dump from the sidebar and click <b>Classify File</b> to run real-time ensemble inference and view threat breakdowns.
                </p>
            </div>
        """, unsafe_allow_html=True)