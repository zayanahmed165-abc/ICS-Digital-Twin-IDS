import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from datetime import datetime


# ============================================================
# ICS DIGITAL TWIN IDS — ADVANCED STREAMLIT DASHBOARD
# Preserves the existing reported model results and adds
# dataset-aware sensor views, filters, exports, and six modules.
# ============================================================

st.set_page_config(
    page_title="ICS Digital Twin IDS | Cybersecurity Console",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------- THEME ---------------------------
BG = "#071426"
PANEL = "#0C2038"
PANEL_2 = "#102B49"
BORDER = "#1C4D70"
TEXT = "#EAF4FF"
MUTED = "#9AB4CE"
CYAN = "#42D9F8"
BLUE = "#2997FF"
GREEN = "#35E0B0"
AMBER = "#FFBF69"
RED = "#FF647C"
PURPLE = "#B58CFF"

st.markdown(
    f"""
    <style>
    :root {{
      color-scheme: dark;
    }}
    .stApp {{
      background:
        radial-gradient(ellipse at 15% 0%, #102D4C 0%, transparent 38%),
        linear-gradient(135deg, {BG} 0%, #081B30 55%, #071426 100%);
      color: {TEXT};
    }}
    [data-testid="stSidebar"] {{
      background: #061326;
      border-right: 1px solid {BORDER};
    }}
    [data-testid="stSidebar"] * {{ color: {TEXT}; }}
    .block-container {{ padding-top: 1.4rem; padding-bottom: 2.4rem; }}
    h1, h2, h3, h4 {{ color: #F4FAFF !important; letter-spacing: -0.02em; }}
    p, label, li, [data-testid="stMarkdownContainer"] {{ color: #D5E7F7; }}
    [data-testid="stMetric"] {{
      background: linear-gradient(145deg, {PANEL_2}, {PANEL});
      border: 1px solid {BORDER};
      padding: 17px 19px;
      border-radius: 15px;
      box-shadow: 0 7px 24px #02081755;
    }}
    [data-testid="stMetricLabel"] {{ color: {MUTED} !important; }}
    [data-testid="stMetricValue"] {{ color: {CYAN} !important; }}
    [data-testid="stDataFrame"] {{
      border: 1px solid {BORDER}; border-radius: 12px; overflow: hidden;
    }}
    .hero {{
      padding: 1.25rem 1.4rem; border-radius: 18px; margin-bottom: 1rem;
      background: linear-gradient(115deg, #0E2B49, #0A1A30 72%);
      border: 1px solid #1B5278;
      box-shadow: 0 10px 32px #02081755;
    }}
    .eyebrow {{
      color: {CYAN}; font-size: .76rem; font-weight: 800;
      letter-spacing: .15em; text-transform: uppercase;
    }}
    .subtle {{ color: {MUTED}; font-size: .9rem; }}
    .status-pill {{
      display: inline-block; border: 1px solid #207C89; color: {CYAN};
      background: #0C3346; padding: 5px 10px; border-radius: 999px;
      font-size: .78rem; font-weight: 700;
    }}
    div.stButton > button, div.stDownloadButton > button {{
      border: 1px solid #2185B6; border-radius: 9px;
      background: linear-gradient(90deg, #087FC4, #1264A2);
      color: white; font-weight: 650;
    }}
    div.stButton > button:hover, div.stDownloadButton > button:hover {{
      border-color: {CYAN}; color: white; background: #0E789F;
    }}
    hr {{ border-color: #1A3E5C; }}
    [data-testid="stAlert"] {{ border-radius: 12px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------- INTEGRATED MODEL RESULTS ----------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
METRICS_FILE = DATA_DIR / "integrated_model_results.csv"
CONFUSION_FILE = DATA_DIR / "integrated_confusion_matrices.csv"

# Safe fallback if evaluation CSV is missing or has an unexpected format.
results = pd.DataFrame({
    "Feature Set": ["ICS Only", "ICS Only", "ICS Only"],
    "Model": ["Random Forest", "Decision Tree", "SVM"],
    "Accuracy": [1.00, 1.00, 1.00],
    "Precision": [1.00, 1.00, 1.00],
    "Recall": [1.00, 1.00, 1.00],
    "F1 Score": [1.00, 1.00, 1.00],
})
integrated_results_loaded = False
metrics_error = None

if METRICS_FILE.exists():
    try:
        loaded_metrics = pd.read_csv(METRICS_FILE)
        loaded_metrics.columns = [str(c).strip() for c in loaded_metrics.columns]
        aliases = {
            "Feature_Set": "Feature Set", "feature_set": "Feature Set",
            "F1": "F1 Score", "F1_Score": "F1 Score",
            "F1 score": "F1 Score", "f1_score": "F1 Score",
        }
        loaded_metrics = loaded_metrics.rename(columns=aliases)
        required_metrics = {"Model", "Accuracy", "Precision", "Recall", "F1 Score"}
        if required_metrics.issubset(loaded_metrics.columns):
            if "Feature Set" not in loaded_metrics.columns:
                loaded_metrics["Feature Set"] = "ICS Only"
            for col in ["Accuracy", "Precision", "Recall", "F1 Score"]:
                loaded_metrics[col] = pd.to_numeric(loaded_metrics[col], errors="coerce")
            loaded_metrics = loaded_metrics.dropna(
                subset=["Accuracy", "Precision", "Recall", "F1 Score"]
            )
            if not loaded_metrics.empty:
                results = loaded_metrics[
                    ["Feature Set", "Model", "Accuracy", "Precision", "Recall", "F1 Score"]
                ].copy()
                integrated_results_loaded = True
        else:
            metrics_error = "Required columns missing from integrated_model_results.csv."
    except Exception as exc:
        metrics_error = str(exc)

# Load saved confusion matrix counts. Format: one row per actual/predicted label pair.
confusion_data = None
confusion_error = None
if CONFUSION_FILE.exists():
    try:
        cm = pd.read_csv(CONFUSION_FILE)
        cm.columns = [str(c).strip() for c in cm.columns]
        required_cm = {"Feature_Set", "Model", "Actual_Label", "Predicted_Label", "Count"}
        if required_cm.issubset(cm.columns):
            cm["Count"] = pd.to_numeric(cm["Count"], errors="coerce").fillna(0).astype(int)
            confusion_data = cm
        else:
            confusion_error = "Required columns missing from integrated_confusion_matrices.csv."
    except Exception as exc:
        confusion_error = str(exc)

# Aggregate multiclass FP/FN: in single-label multiclass classification, each wrong
# prediction contributes one false positive for its predicted class and one false negative
# for its actual class. These are aggregate counts, not class-specific counts.
fp_fn_rows = []
if confusion_data is not None:
    for (feature_set, model), group in confusion_data.groupby(["Feature_Set", "Model"]):
        errors = int(group.loc[group["Actual_Label"] != group["Predicted_Label"], "Count"].sum())
        fp_fn_rows.append({
            "Feature Set": feature_set,
            "Model": model,
            "False Positives": errors,
            "False Negatives": errors,
        })
fp_fn = pd.DataFrame(fp_fn_rows, columns=[
    "Feature Set", "Model", "False Positives", "False Negatives"
])

# Keep the original reported noise experiment clearly separate from integrated metrics.
robustness = pd.DataFrame({
    "Model": ["Random Forest", "SVM", "Isolation Forest"],
    "Original Accuracy": [1.00, 0.9730769231, 0.7346153846],
    "Noisy Accuracy": [1.00, 0.9730769231, 0.7307692308],
})

# -------------------------- DATA LOADING ----------------------
ROOT = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    ROOT / "data" / "integrated_dataset.csv",
    ROOT / "data" / "final_dataset.csv",
    ROOT / "final_dataset.csv",
    ROOT / "data" / "digital_twin_data.csv",
    ROOT / "digital_twin_data.csv",
]
RESULTS_DIR = ROOT / "results"

@st.cache_data(show_spinner=False)
def load_csv(path_string, modified_time):
    # modified_time is included in the cache key so edits to the CSV refresh it.
    del modified_time
    return pd.read_csv(path_string)

dataset = None
dataset_path = None
load_error = None
for candidate in DATA_CANDIDATES:
    if candidate.exists():
        try:
            dataset = load_csv(str(candidate), candidate.stat().st_mtime)
            dataset_path = candidate
            break
        except Exception as exc:
            load_error = str(exc)

def find_label_column(df):
    if df is None:
        return None
    exact = ["label", "target", "class", "attack", "status", "y"]
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for name in exact:
        if name in lower_map:
            return lower_map[name]
    return None

def safe_num(series):
    return pd.to_numeric(series, errors="coerce")

label_col = find_label_column(dataset)
feature_columns = []
if dataset is not None:
    preferred = [
        "temperature", "pressure", "flow_rate", "tank_level",
        "cpu_usage", "ram_usage", "disk_usage", "process_count"
    ]
    lower_map = {str(c).strip().lower(): c for c in dataset.columns}
    feature_columns = [lower_map[x] for x in preferred if x in lower_map]
    if not feature_columns:
        feature_columns = [
            c for c in dataset.select_dtypes(include="number").columns
            if c != label_col
        ]

record_count = len(dataset) if dataset is not None else 1300
if dataset is not None and label_col:
    label_counts = dataset[label_col].astype(str).value_counts(dropna=False)
else:
    label_counts = pd.Series(dtype=int)

# ---------------------------- SIDEBAR -------------------------
st.sidebar.markdown("# 🛡️ ICS DIGITAL TWIN")
st.sidebar.caption("INTRUSION DETECTION CONSOLE")
st.sidebar.markdown(
    '<span class="status-pill">SIMULATION / EVALUATION MODE</span>',
    unsafe_allow_html=True,
)
st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview",
        "Model Comparison",
        "FP / FN Analysis",
        "Digital Twin Monitoring",
        "Robustness Evaluation",
        "Threat Logs & Reports",
        "Ablation Study",
        "About",
    ],
)
st.sidebar.divider()
st.sidebar.markdown("**SYSTEM SNAPSHOT**")
st.sidebar.caption(f"Dataset records: {record_count:,}")
st.sidebar.caption(f"Dataset file: {dataset_path.name if dataset_path else 'Not found'}")
st.sidebar.caption("Models: Random Forest • SVM • Isolation Forest")
st.sidebar.caption("Metrics are simulation-based; no live ICS connection is implied.")

# ----------------------------- HEADER ------------------------
st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Industrial OT Security • ML Analytics</div>
      <h1 style="margin:.35rem 0 .15rem 0;">🛡️ ICS Digital Twin IDS</h1>
      <div style="font-size:1.05rem;font-weight:650;color:#C9E7FF;">
        Machine Learning Model Evaluation & Threat Monitoring
      </div>
      <p class="subtle" style="margin:.55rem 0 0 0;">
        Simulation-based intrusion detection • Dataset-aware monitoring • Model evaluation
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if dataset is None:
    st.warning(
        "Dataset CSV nahi mili. Model evaluation pages reported metrics ke saath kaam karengi, "
        "lekin sensor charts aur dataset-derived threat rows ke liye "
        "`data/integrated_dataset.csv` ya `data/final_dataset.csv` file project ke andar honi chahiye."
    )
    if load_error:
        st.caption(f"CSV read error: {load_error}")
else:
    st.caption(f"Loaded dataset: `{dataset_path.relative_to(ROOT)}` • {len(dataset):,} rows • {dataset.shape[1]} columns")

# ------------------------- CHART HELPERS ----------------------
def style_axes(ax, ylabel=None, ylim=None):
    ax.set_facecolor(PANEL)
    ax.figure.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED)
    if ylim:
        ax.set_ylim(*ylim)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    ax.grid(axis="y", alpha=0.18, color="#8ABDE5")
    ax.title.set_color(TEXT)

def show_model_metrics():
    st.subheader("Integrated model performance")
    if integrated_results_loaded:
        st.caption("Metrics loaded from data/integrated_model_results.csv.")
    else:
        st.warning("Integrated metrics CSV load nahi hui; fallback values display ho rahi hain.")
        if metrics_error:
            st.caption(f"Metrics loading detail: {metrics_error}")
    metric_view = results.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1 Score"]:
        metric_view[col] = pd.to_numeric(metric_view[col], errors="coerce").map(
            lambda value: f"{value:.2%}" if pd.notna(value) else "N/A"
        )
    st.dataframe(metric_view, use_container_width=True, hide_index=True)

def show_accuracy_chart():
    chart = results.copy()
    chart["Comparison"] = chart["Feature Set"].astype(str) + " — " + chart["Model"].astype(str)
    colors = [GREEN, BLUE, PURPLE, CYAN, AMBER, RED]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    bars = ax.bar(
        chart["Comparison"], chart["Accuracy"] * 100,
        color=[colors[i % len(colors)] for i in range(len(chart))], width=0.62
    )
    style_axes(ax, "Accuracy (%)", (0, 110))
    ax.set_title("Model accuracy by feature set", loc="left", pad=12, fontweight="bold")
    ax.tick_params(axis="x", labelrotation=25)
    for bar, value in zip(bars, chart["Accuracy"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(bar.get_height() + 2, 107),
            f"{value:.1%}", ha="center", color=TEXT, fontweight="bold", fontsize=8
        )
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

def show_confusion_matrix():
    st.subheader("Confusion matrix")
    if confusion_data is None or confusion_data.empty:
        st.info("Confusion matrix CSV nahi mili ya format match nahi hua.")
        if confusion_error:
            st.caption(confusion_error)
        return
    feature_sets = sorted(confusion_data["Feature_Set"].astype(str).unique())
    selected_set = st.selectbox("Feature set", feature_sets, key="cm_feature_set")
    models = sorted(confusion_data.loc[
        confusion_data["Feature_Set"].astype(str) == selected_set, "Model"
    ].astype(str).unique())
    selected_model = st.selectbox("Model", models, key="cm_model")
    part = confusion_data[
        (confusion_data["Feature_Set"].astype(str) == selected_set) &
        (confusion_data["Model"].astype(str) == selected_model)
    ]
    matrix = part.pivot_table(
        index="Actual_Label", columns="Predicted_Label", values="Count",
        aggfunc="sum", fill_value=0
    )
    labels = sorted(set(matrix.index.astype(str)) | set(matrix.columns.astype(str)))
    matrix = matrix.reindex(index=labels, columns=labels, fill_value=0)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    fig.patch.set_facecolor(PANEL)
    ax.set_facecolor(PANEL)
    image = ax.imshow(matrix.values, cmap="Blues")
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.set_xlabel("Predicted label", color=MUTED)
    ax.set_ylabel("Actual label", color=MUTED)
    ax.set_title(f"{selected_model} — {selected_set}", color=TEXT, loc="left", pad=12)
    ax.tick_params(colors=TEXT)
    threshold = matrix.values.max() / 2 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = int(matrix.iloc[i, j])
            ax.text(j, i, str(value), ha="center", va="center",
                    color="white" if value > threshold else "#071426", fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.download_button(
        "⬇️ Download selected confusion matrix CSV",
        part.to_csv(index=False).encode("utf-8"),
        file_name="ics_selected_confusion_matrix.csv", mime="text/csv"
    )

def display_dataset_preview(df, rows=12):
    st.dataframe(df.head(rows), use_container_width=True, hide_index=True)

# ---------------------------- OVERVIEW ------------------------
if page == "Overview":
    st.markdown("### Operational overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset records", f"{record_count:,}")
    c2.metric("Model evaluations", f"{len(results):,}")
    best_row = results.loc[results["Accuracy"].idxmax()]
    c3.metric("Top saved accuracy", f"{best_row['Accuracy']:.1%}", str(best_row["Model"]))
    c4.metric("Sensor features found", str(len(feature_columns)) if dataset is not None else "Awaiting CSV")

    left, right = st.columns([1.35, 1])
    with left:
        show_accuracy_chart()
    with right:
        st.markdown("#### Dataset profile")
        if dataset is not None:
            st.metric("Columns", f"{dataset.shape[1]}")
            st.metric("Missing cells", f"{int(dataset.isna().sum().sum()):,}")
            st.metric("Duplicate rows", f"{int(dataset.duplicated().sum()):,}")
            if label_col:
                st.markdown(f"**Detected target column:** `{label_col}`")
                st.dataframe(
                    label_counts.rename_axis("Label").reset_index(name="Records"),
                    use_container_width=True, hide_index=True
                )
            else:
                st.info("Label column auto-detect nahi hui. Dataset columns check karein.")
        else:
            st.info("Dataset profile ke liye `data/final_dataset.csv` load karein.")

    st.divider()
    show_model_metrics()
    st.info(
        "Research note: Random Forest ka 100% reported accuracy simulated test split ka result hai. "
        "Independent data, leakage checks aur repeated evaluation ke baghair ise real-world guarantee na samjhein."
    )

# ------------------------ MODEL COMPARISON --------------------
elif page == "Model Comparison":
    st.markdown("### Model comparison lab")
    selected_metrics = st.multiselect(
        "Metrics to compare",
        ["Accuracy", "Precision", "Recall", "F1 Score"],
        default=["Accuracy", "Precision", "Recall", "F1 Score"],
    )
    if selected_metrics:
        chart_data = results.copy()
        chart_data["Comparison"] = chart_data["Feature Set"].astype(str) + " — " + chart_data["Model"].astype(str)
        chart_df = chart_data.set_index("Comparison")[selected_metrics]
        st.bar_chart(chart_df, y_label="Score (0–1)")
    show_model_metrics()
    st.download_button(
        "⬇️ Download model metrics CSV",
        results.to_csv(index=False).encode("utf-8"),
        file_name="ics_model_metrics.csv",
        mime="text/csv",
    )
    st.caption("Displayed metrics are loaded from saved evaluation outputs; no model is retrained by the dashboard.")
    show_confusion_matrix()

# ------------------------- FP / FN ANALYSIS -------------------
elif page == "FP / FN Analysis":
    st.markdown("### 🚨 False positive / false negative analysis")
    st.caption(
        "Counts are aggregated from the saved multiclass confusion matrices. Each incorrect prediction "
        "contributes one FP for the predicted class and one FN for the actual class; class-specific detail "
        "is available in the confusion matrix view."
    )
    if fp_fn.empty:
        st.info("FP/FN analysis requires data/integrated_confusion_matrices.csv.")
    else:
        st.dataframe(fp_fn, use_container_width=True, hide_index=True)
        plot_data = fp_fn.copy()
        plot_data["Comparison"] = plot_data["Feature Set"].astype(str) + " — " + plot_data["Model"].astype(str)
        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(len(plot_data))
        width = 0.36
        ax.bar(x - width/2, plot_data["False Positives"], width, label="Aggregate false positives", color=AMBER)
        ax.bar(x + width/2, plot_data["False Negatives"], width, label="Aggregate false negatives", color=RED)
        ax.set_xticks(x, plot_data["Comparison"], rotation=25, ha="right")
        style_axes(ax, "Count")
        ax.set_title("Aggregate error counts by feature set and model", loc="left", pad=12, fontweight="bold")
        ax.legend(facecolor=PANEL_2, labelcolor=TEXT, frameon=True)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.download_button(
            "⬇️ Download FP / FN CSV",
            fp_fn.to_csv(index=False).encode("utf-8"),
            file_name="ics_fp_fn_analysis.csv",
            mime="text/csv",
        )
        show_confusion_matrix()

# -------------------- DIGITAL TWIN MONITORING -----------------
elif page == "Digital Twin Monitoring":
    st.markdown("### 🏭 Digital twin sensor monitoring")
    st.caption("Charts are generated only from columns found in your dataset; this is not a live plant connection.")
    if dataset is None:
        st.info("Dataset load nahi hui. `data/integrated_dataset.csv` ko project folder mein check karein.")
    elif not feature_columns:
        st.warning("CSV mein recognized numeric sensor features nahi mile.")
        st.write("Available columns:", list(dataset.columns))
    else:
        chosen = st.multiselect(
            "Select sensor / telemetry features",
            feature_columns,
            default=feature_columns[:min(4, len(feature_columns))],
        )
        if chosen:
            numeric_data = dataset[chosen].apply(pd.to_numeric, errors="coerce")
            stats = pd.DataFrame({
                "Feature": chosen,
                "Mean": [numeric_data[c].mean() for c in chosen],
                "Minimum": [numeric_data[c].min() for c in chosen],
                "Maximum": [numeric_data[c].max() for c in chosen],
                "Missing": [int(numeric_data[c].isna().sum()) for c in chosen],
            })
            st.dataframe(
                stats.style.format({
                    "Mean": "{:.3f}", "Minimum": "{:.3f}", "Maximum": "{:.3f}"
                }),
                use_container_width=True, hide_index=True
            )
            limit = st.slider("Rows to display in trend", min_value=50, max_value=max(50, min(1300, len(dataset))), value=min(300, max(50, len(dataset))), step=50)
            st.line_chart(numeric_data[chosen].head(limit), y_label="Recorded value")
            st.download_button(
                "⬇️ Download selected sensor data",
                numeric_data[chosen].to_csv(index=False).encode("utf-8"),
                file_name="ics_sensor_monitoring.csv",
                mime="text/csv",
            )
        st.markdown("#### Dataset quality")
        quality = pd.DataFrame({
            "Column": dataset.columns.astype(str),
            "Data type": [str(t) for t in dataset.dtypes],
            "Missing values": [int(dataset[c].isna().sum()) for c in dataset.columns],
            "Unique values": [int(dataset[c].nunique(dropna=True)) for c in dataset.columns],
        })
        st.dataframe(quality, use_container_width=True, hide_index=True)

# ------------------------ ROBUSTNESS --------------------------
elif page == "Robustness Evaluation":
    st.markdown("### 🧪 Robustness evaluation")
    st.write("Original accuracy versus accuracy after the reported Gaussian-noise experiment (standard deviation 0.05).")
    st.dataframe(
        robustness.style.format({
            "Original Accuracy": "{:.2%}",
            "Noisy Accuracy": "{:.2%}",
        }),
        use_container_width=True, hide_index=True,
    )
    chart_df = robustness.set_index("Model")
    st.bar_chart(chart_df, y_label="Accuracy (0–1)")
    robustness["Accuracy Change"] = robustness["Noisy Accuracy"] - robustness["Original Accuracy"]
    st.markdown("#### Change after noise")
    st.dataframe(
        robustness[["Model", "Accuracy Change"]].style.format({"Accuracy Change": "{:+.2%}"}),
        use_container_width=True, hide_index=True,
    )
    st.download_button(
        "⬇️ Download robustness CSV",
        robustness.to_csv(index=False).encode("utf-8"),
        file_name="ics_robustness_results.csv",
        mime="text/csv",
    )
    st.caption("This view displays the reported experiment; it does not rerun model predictions.")

# ---------------------- THREAT LOGS / REPORTS -----------------
elif page == "Threat Logs & Reports":
    st.markdown("### 📋 Dataset records & report exports")
    if dataset is not None:
        st.caption("These are dataset rows, not confirmed model-generated alerts. No prediction column is fabricated.")
        view = dataset.copy()
        if label_col:
            labels = sorted(view[label_col].astype(str).dropna().unique().tolist())
            selected_labels = st.multiselect("Filter by dataset label", labels, default=labels)
            if selected_labels:
                view = view[view[label_col].astype(str).isin(selected_labels)]
        search = st.text_input("Search rows (any column)", placeholder="Enter a value to filter records...")
        if search:
            mask = view.astype(str).apply(
                lambda col: col.str.contains(search, case=False, na=False, regex=False)
            ).any(axis=1)
            view = view[mask]
        st.metric("Rows matching filters", f"{len(view):,}")
        st.dataframe(view.head(1000), use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Export filtered dataset as CSV",
            view.to_csv(index=False).encode("utf-8"),
            file_name="ics_filtered_dataset_records.csv",
            mime="text/csv",
        )
    else:
        st.info("Dataset records show karne ke liye `data/integrated_dataset.csv` add karein.")

    report_summary = pd.DataFrame({
        "Item": [
            "Generated at", "Dataset file", "Dataset rows", "Dataset columns",
            "Models evaluated", "Best reported model", "Best reported accuracy",
            "Important limitation",
        ],
        "Value": [
            datetime.now().isoformat(timespec="seconds"),
            str(dataset_path.relative_to(ROOT)) if dataset_path else "Not found",
            str(len(dataset)) if dataset is not None else "Not available",
            str(dataset.shape[1]) if dataset is not None else "Not available",
            "; ".join(sorted(results["Model"].astype(str).unique())),
            str(results.loc[results["Accuracy"].idxmax(), "Model"]),
            f"{results['Accuracy'].max():.1%} (saved evaluation result)",
            "Simulated dataset; validate on independent data",
        ],
    })
    st.markdown("#### Evaluation summary")
    st.dataframe(report_summary, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download evaluation summary CSV",
        report_summary.to_csv(index=False).encode("utf-8"),
        file_name="ics_evaluation_summary.csv",
        mime="text/csv",
    )

# -------------------------- ABLATION --------------------------
elif page == "Ablation Study":
    st.markdown("### ⚙️ Feature ablation study")
    ablation = pd.DataFrame({
        "Experiment": ["Baseline Random Forest", "Without cpu_usage"],
        "Features": ["All 8 features", "7 features"],
        "Accuracy": [1.00, 1.00],
        "Change vs baseline": [0.00, 0.00],
    })
    a1, a2, a3 = st.columns(3)
    a1.metric("Feature removed", "cpu_usage")
    a2.metric("Baseline accuracy", "100%")
    a3.metric("Accuracy change", "0.00 percentage points")
    st.dataframe(
        ablation.style.format({"Accuracy": "{:.2%}", "Change vs baseline": "{:+.2%}"}),
        use_container_width=True, hide_index=True,
    )
    st.info(
        "Reported result: removing cpu_usage did not change Random Forest accuracy on this split. "
        "This does not prove the feature is unimportant generally."
    )
    st.download_button(
        "⬇️ Download ablation CSV",
        ablation.to_csv(index=False).encode("utf-8"),
        file_name="ics_ablation_study.csv",
        mime="text/csv",
    )

# ----------------------------- ABOUT --------------------------
elif page == "About":
    st.markdown("### ℹ️ About this project")
    st.markdown("""
    This prototype compares machine-learning approaches for intrusion detection on a **simulated
    Industrial Control System (ICS)** dataset.

    **Models**
    - **Random Forest:** supervised classification.
    - **Support Vector Machine (SVM):** supervised classification.
    - **Isolation Forest:** unsupervised anomaly detection.

    **Dataset features (when available)**
    - Process telemetry: temperature, pressure, flow rate and tank level.
    - Host telemetry: CPU usage, RAM usage, disk usage and process count.

    **Evaluation modules**
    - Model performance comparison.
    - False-positive / false-negative analysis.
    - Dataset-aware sensor trend monitoring.
    - Noise robustness comparison.
    - Feature ablation study.
    - CSV exports and evaluation summary.
    """)
    st.warning(
        "This is not connected to a live industrial control system. All performance claims must be "
        "validated against actual saved model outputs and independent data before operational use."
    )
    st.markdown("#### Expected project files")
    st.code(
        "ICS_Digital_Twin_IDS/\n"
        "├── dashboard_advanced.py\n"
        "├── data/\n"
        "│   ├── integrated_dataset.csv\n"
        "│   ├── integrated_model_results.csv\n"
        "│   └── integrated_confusion_matrices.csv\n"
        "├── src/\n"
        "│   └── ml_models.py\n"
        "├── models/       (optional saved model files)\n"
        "└── results/      (optional saved evaluation outputs)",
        language="text",
    )

# ----------------------------- FOOTER -------------------------
st.divider()
st.markdown(
    '<p class="subtle">ICS Digital Twin IDS • Streamlit • Simulation-based research prototype</p>',
    unsafe_allow_html=True,
)
