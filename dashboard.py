
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="ICS Digital Twin IDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# DARK THEME CSS
# =========================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #071426, #0B1D35);
    color: #F1F5F9;
}
[data-testid="stSidebar"] {
    background: #09172B;
    border-right: 1px solid #183B5B;
}
[data-testid="stSidebar"] * {
    color: #E2E8F0;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
h1, h2, h3 {
    color: #F8FAFC !important;
}
p, label, [data-testid="stMarkdownContainer"] {
    color: #D5E3F3;
}
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #102B49, #0D2039);
    border: 1px solid #17618A;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0 4px 15px #02081755;
}
div[data-testid="stMetricLabel"] {
    color: #9FC7EB !important;
}
div[data-testid="stMetricValue"] {
    color: #36E0B0 !important;
}
div[data-testid="stDataFrame"] {
    border: 1px solid #1B4568;
    border-radius: 12px;
    overflow: hidden;
}
hr {
    border-color: #1B4568;
}
div.stButton > button {
    background: #087FC4;
    color: white;
    border: none;
    border-radius: 8px;
}
div.stButton > button:hover {
    background: #12B8A6;
    color: #071426;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.sidebar.markdown("# 🛡️ ICS Digital Twin")
st.sidebar.markdown("### Intrusion Detection System")
st.sidebar.caption("Machine Learning • Cybersecurity • Digital Twin")
st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview",
        "Model Comparison",
        "FP / FN Analysis",
        "Robustness Evaluation",
        "Ablation Study",
        "About"
    ]
)

st.title("🛡️ ICS Digital Twin IDS")
st.markdown("### Machine Learning Model Evaluation Dashboard")
st.caption("Industrial Control System • Simulated Dataset • 1,300 Records")

st.divider()

# =========================
# DATA
# =========================
results = pd.DataFrame({
    "Model": ["Random Forest", "SVM", "Isolation Forest"],
    "Accuracy": [1.00, 0.9730769231, 0.7346153846],
    "Precision": [1.00, 0.98, 0.73],
    "Recall": [1.00, 0.94, 0.83],
    "F1 Score": [1.00, 0.96, 0.71]
})

fp_fn = pd.DataFrame({
    "Model": ["Random Forest", "SVM", "Isolation Forest"],
    "False Positives": [0, 0, 69],
    "False Negatives": [0, 7, 0]
})

robustness = pd.DataFrame({
    "Model": ["Random Forest", "SVM", "Isolation Forest"],
    "Original Accuracy": [1.00, 0.9730769231, 0.7346153846],
    "Noisy Accuracy": [1.00, 0.9730769231, 0.7307692308]
})

# =========================
# OVERVIEW
# =========================
if page in ["Overview", "Model Comparison"]:

    st.subheader("📊 Model Performance")

    c1, c2, c3 = st.columns(3)

    c1.metric("Best Accuracy", "100%", "Random Forest")
    c2.metric("Best Model", "Random Forest")
    c3.metric("Dataset Records", "1,300")

    st.markdown("#### Performance Metrics")

    display_results = results.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1 Score"]:
        display_results[col] = display_results[col].map(
            lambda x: f"{x:.2%}"
        )

    st.dataframe(
        display_results,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("#### 🎯 Accuracy Comparison")

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#0B1D35")
    ax.set_facecolor("#0B1D35")

    bars = ax.bar(
        results["Model"],
        results["Accuracy"] * 100,
        color=["#19D3A2", "#2997FF", "#A855F7"],
        width=0.58
    )

    ax.set_ylabel("Accuracy (%)", color="#D5E3F3")
    ax.set_ylim(0, 110)
    ax.tick_params(colors="#D5E3F3")
    ax.spines["bottom"].set_color("#31516E")
    ax.spines["left"].set_color("#31516E")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.18, color="#A9C7E8")

    for bar, value in zip(bars, results["Accuracy"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{value:.2%}",
            ha="center",
            color="white",
            fontweight="bold"
        )

    st.pyplot(fig)
    plt.close(fig)

    st.info(
        "Random Forest achieved the highest accuracy on this "
        "simulated test dataset. Validate the result on independent "
        "data before using it in a real industrial system."
    )

# =========================
# FALSE POSITIVE / NEGATIVE
# =========================
if page in ["Overview", "FP / FN Analysis"]:

    st.divider()
    st.subheader("🚨 False Positive / False Negative Analysis")

    st.caption(
        "False positive: normal activity flagged as an attack. "
        "False negative: an attack missed by the model."
    )

    st.dataframe(
        fp_fn,
        use_container_width=True,
        hide_index=True
    )

    fig, ax = plt.subplots(figsize=(9, 3.5))
    fig.patch.set_facecolor("#0B1D35")
    ax.set_facecolor("#0B1D35")

    x = range(len(fp_fn))
    width = 0.34

    ax.bar(
        [i - width / 2 for i in x],
        fp_fn["False Positives"],
        width,
        label="False Positives",
        color="#FFB020"
    )
    ax.bar(
        [i + width / 2 for i in x],
        fp_fn["False Negatives"],
        width,
        label="False Negatives",
        color="#FF5D73"
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(fp_fn["Model"], color="#D5E3F3")
    ax.tick_params(axis="y", colors="#D5E3F3")
    ax.set_ylabel("Count", color="#D5E3F3")
    ax.legend(facecolor="#102B49", labelcolor="white")
    ax.grid(axis="y", alpha=0.15)

    for spine in ax.spines.values():
        spine.set_color("#31516E")

    st.pyplot(fig)
    plt.close(fig)

# =========================
# ROBUSTNESS
# =========================
if page in ["Overview", "Robustness Evaluation"]:

    st.divider()
    st.subheader("🧪 Robustness Evaluation")
    st.write(
        "Comparison of original model accuracy with accuracy "
        "after adding Gaussian noise (standard deviation 0.05)."
    )

    st.dataframe(
        robustness.style.format({
            "Original Accuracy": "{:.2%}",
            "Noisy Accuracy": "{:.2%}"
        }),
        use_container_width=True
    )

    fig, ax = plt.subplots(figsize=(9, 3.5))
    fig.patch.set_facecolor("#0B1D35")
    ax.set_facecolor("#0B1D35")

    robustness.plot(
        x="Model",
        y=["Original Accuracy", "Noisy Accuracy"],
        kind="bar",
        ax=ax,
        color=["#2997FF", "#19D3A2"],
        width=0.72
    )

    ax.set_ylabel("Accuracy", color="#D5E3F3")
    ax.set_ylim(0, 1.1)
    ax.tick_params(colors="#D5E3F3")
    ax.legend(facecolor="#102B49", labelcolor="white")
    ax.grid(axis="y", alpha=0.15)

    for spine in ax.spines.values():
        spine.set_color("#31516E")

    st.pyplot(fig)
    plt.close(fig)

# =========================
# ABLATION STUDY
# =========================
if page in ["Overview", "Ablation Study"]:

    st.divider()
    st.subheader("⚙️ Ablation Study")

    a1, a2 = st.columns(2)
    a1.metric("Feature Removed", "cpu_usage")
    a2.metric("Accuracy Change", "0.00")

    st.success(
        "Removing cpu_usage did not change Random Forest accuracy "
        "in this experiment. This does not prove the feature is "
        "unimportant on other datasets."
    )

# =========================
# ABOUT
# =========================
if page == "About":

    st.subheader("ℹ️ About This Project")

    st.markdown("""
    This project evaluates machine learning approaches for a
    simulated Industrial Control System (ICS) intrusion detection task.

    **Models evaluated**
    - Random Forest — supervised classification
    - Support Vector Machine (SVM) — supervised classification
    - Isolation Forest — anomaly detection

    **Dataset features**
    - Temperature, pressure, flow rate and tank level
    - CPU usage, RAM usage, disk usage and process count

    **Evaluation**
    - Accuracy, precision, recall and F1 score
    - False positives and false negatives
    - Noise robustness experiment
    - Feature ablation experiment

    **Important:** These results come from a simulated dataset.
    They should not be interpreted as proof of real-world
    industrial security performance.
    """)

st.sidebar.divider()
st.sidebar.caption("ICS Digital Twin IDS • Built with Streamlit")
