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


st.set_page_config(
    page_title="Cyber Anomaly Detector",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 2rem; max-width: 1000px; }
    .doc-status {
        padding: 0.6rem 0.8rem; border-radius: 8px;
        background-color: rgba(46, 204, 113, 0.12);
        border: 1px solid rgba(46, 204, 113, 0.4);
        font-size: 0.85rem; margin-top: 0.5rem;
    }
    .attack-status {
        padding: 0.6rem 0.8rem; border-radius: 8px;
        background-color: rgba(231, 76, 60, 0.12);
        border: 1px solid rgba(231, 76, 60, 0.4);
        font-size: 0.85rem; margin-top: 0.5rem;
    }
    .stButton button { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_pipeline_artifacts():
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


@st.cache_data
def load_test_set():
    X_test = pd.read_csv(f"{PROCESSED_DIR}/X_test.csv")
    y_test_binary = pd.read_csv(f"{PROCESSED_DIR}/y_test_binary.csv").squeeze()
    y_test_multi = pd.read_csv(f"{PROCESSED_DIR}/y_test_multiclass.csv").squeeze()
    return X_test, y_test_binary, y_test_multi


def strip_label_columns(df: pd.DataFrame):
    cols_found = [c for c in df.columns if c in LABEL_COLUMNS_TO_STRIP]
    if cols_found:
        df = df.drop(columns=cols_found)
    return df, cols_found


def classify_rows(X_scaled, rf_binary, xgb_binary, rf_multi, encoder):
    """
    Decision rule: a row is flagged as Attack based on the RANDOM FOREST
    prediction (matches the single-row demo tab's behavior, so the app is
    consistent). XGBoost's prediction is shown alongside as a secondary
    reference only - it does not affect flagging.
    """
    rf_preds = rf_binary.predict(X_scaled)
    xgb_preds = xgb_binary.predict(X_scaled)

    results = pd.DataFrame({
        "Random Forest": pd.Series(rf_preds).map({0: "Benign", 1: "Attack"}).values,
        "XGBoost (reference)": pd.Series(xgb_preds).map({0: "Benign", 1: "Attack"}).values,
    })

    if rf_multi is not None:
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


# -session state

if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_source" not in st.session_state:
    st.session_state.last_source = None
if "rows_classified" not in st.session_state:
    st.session_state.rows_classified = 0


scaler, feature_columns, rf_binary, xgb_binary, rf_multi, encoder = load_pipeline_artifacts()


# Sidebar

with st.sidebar:
    st.markdown("Cyber Anomaly Detector")
    st.caption("Clustering (K-Means + Isolation Forest) generated training "
               "labels — Random Forest / XGBoost classify new traffic below.")
    st.divider()

    uploaded_file = st.file_uploader("Upload new traffic (CSV)", type="csv")

    load_col, clear_col = st.columns([2, 1])
    with load_col:
        load_clicked = st.button("Classify File", use_container_width=True,
                                  type="primary", disabled=uploaded_file is None)
    with clear_col:
        if st.button("Reset", use_container_width=True):
            st.session_state.last_result = None
            st.session_state.last_source = None
            st.session_state.rows_classified = 0
            st.rerun()

    if uploaded_file and load_clicked:
        progress = st.progress(0, text="Reading CSV...")
        with timer("Read CSV"):
            raw_df = pd.read_csv(uploaded_file, low_memory=False)
            raw_df.columns = [c.strip() for c in raw_df.columns]

        progress.progress(20, text="Checking for label columns...")
        raw_df, stripped_cols = strip_label_columns(raw_df)
        if stripped_cols:
            st.toast(f"Removed label column(s) found in upload: {stripped_cols} "
                      f"- these are never used for prediction.")

        progress.progress(35, text="Cleaning data...")
        with timer("Clean"):
            raw_df = raw_df.replace([np.inf, -np.inf], np.nan).dropna()

        missing = set(feature_columns) - set(raw_df.columns)
        if missing:
            progress.empty()
            st.error(f"Missing required columns: {missing}")
        else:
            progress.progress(55, text="Scaling features...")
            with timer("Scale"):
                X_new = raw_df[feature_columns]
                X_scaled = pd.DataFrame(
                    scaler.transform(X_new), columns=X_new.columns, index=X_new.index
                )

            progress.progress(80, text="Classifying...")
            with timer("Classify"):
                result = classify_rows(X_scaled, rf_binary, xgb_binary, rf_multi, encoder)

            st.session_state.last_result = result
            st.session_state.last_source = uploaded_file.name
            st.session_state.rows_classified = len(result)

            progress.progress(100, text="Done")
            st.rerun()

    if st.session_state.last_source:
        n_attacks = (st.session_state.last_result["Random Forest"] == "Attack").sum()
        box_class = "attack-status" if n_attacks > 0 else "doc-status"
        st.markdown(
            f'<div class="{box_class}"><b>{st.session_state.last_source}</b><br>'
            f'{st.session_state.rows_classified} rows classified · '
            f'{n_attacks} flagged as attack</div>',
            unsafe_allow_html=True
        )
    else:
        st.info("No file classified yet.")

    if st.session_state.get("timings"):
        with st.expander("Timing breakdown"):
            for label, secs in st.session_state.timings[-8:]:
                st.write(f"{label}: {secs:.3f}s")

    st.divider()
    st.caption("Or try a real example from the held-out test set below.")


# main area

st.title("Live Traffic Classification")

tab1, tab2 = st.tabs(["Try a Real Example", "Uploaded File Results"])

with tab1:
    st.subheader("Pick a random row from the real, held-out test set")
    X_test, y_test_binary, y_test_multi = load_test_set()

    if st.button("Try a Random Example"):
        idx = X_test.sample(1).index[0]
        row = X_test.loc[[idx]]
        true_binary = "Attack" if y_test_binary.loc[idx] == 1 else "Benign"
        true_multi = y_test_multi.loc[idx]

        st.write(f"**True label:** {true_binary} ({true_multi})")

        with timer("Random example classify"):
            result = classify_rows(row, rf_binary, xgb_binary, rf_multi, encoder)
        st.dataframe(result, use_container_width=True)

with tab2:
    st.subheader("Results from your uploaded file")
    if st.session_state.last_result is not None:
        result = st.session_state.last_result
        total, n_normal, n_attack, breakdown = summarize_results(result)

        st.write(f"Source: **{st.session_state.last_source}**")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total flows analyzed", f"{total:,}")
        col2.metric("Normal", f"{n_normal:,}")
        col3.metric("Flagged as attack", f"{n_attack:,}")

        st.caption(
            "These are model predictions based on the trained pipeline, "
            "not confirmed incidents - treat flagged rows as candidates "
            "for further investigation, not verified attacks."
        )

        if breakdown is not None and len(breakdown) > 0:
            st.write("**Attack-type breakdown (flagged rows only):**")
            st.bar_chart(breakdown)
            st.dataframe(breakdown.rename("Count"), use_container_width=True)

        st.write("**Per-row predictions:**")
        st.dataframe(result, use_container_width=True)

        st.download_button(
            "Download predictions as CSV",
            result.to_csv(index=False),
            file_name="predictions.csv",
            mime="text/csv",
        )
    else:
        st.markdown("Upload a CSV from the sidebar to see results here.")