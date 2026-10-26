
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

DATA_FILE = os.path.join(DATA_DIR, "integrated_dataset.csv")
RESULTS_FILE = os.path.join(DATA_DIR, "integrated_model_results.csv")
REPORT_FILE = os.path.join(DATA_DIR, "integrated_classification_reports.txt")
MATRIX_FILE = os.path.join(DATA_DIR, "integrated_confusion_matrices.csv")

data = pd.read_csv(DATA_FILE)

ics_features = [
    "temperature", "pressure", "flow_rate", "tank_level"
]
os_features = [
    "cpu_usage", "ram_usage", "disk_usage", "process_count"
]

y = data["label"]
labels = sorted(y.unique())

# Use the same row split for both feature sets
train_idx, test_idx = train_test_split(
    data.index,
    test_size=0.20,
    random_state=42,
    stratify=y
)

feature_sets = {
    "ICS Only": ics_features,
    "ICS + OS Telemetry": ics_features + os_features
}

def make_models():
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced"
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced"
        ),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(
                kernel="rbf",
                random_state=42,
                class_weight="balanced"
            ))
        ])
    }

results = []
matrix_rows = []
report_sections = []

for feature_set_name, features in feature_sets.items():
    X_train = data.loc[train_idx, features]
    X_test = data.loc[test_idx, features]
    y_train = y.loc[train_idx]
    y_test = y.loc[test_idx]

    for model_name, model in make_models().items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        results.append({
            "Feature_Set": feature_set_name,
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision_Weighted": precision_score(
                y_test, predictions,
                average="weighted", zero_division=0
            ),
            "Recall_Weighted": recall_score(
                y_test, predictions,
                average="weighted", zero_division=0
            ),
            "F1_Weighted": f1_score(
                y_test, predictions,
                average="weighted", zero_division=0
            )
        })

        report = classification_report(
            y_test, predictions,
            labels=labels, zero_division=0
        )

        report_sections.append(
            f"FEATURE SET: {feature_set_name}\n"
            f"MODEL: {model_name}\n{report}\n"
        )

        matrix = confusion_matrix(
            y_test, predictions, labels=labels
        )

        for i, actual_label in enumerate(labels):
            for j, predicted_label in enumerate(labels):
                matrix_rows.append({
                    "Feature_Set": feature_set_name,
                    "Model": model_name,
                    "Actual_Label": actual_label,
                    "Predicted_Label": predicted_label,
                    "Count": int(matrix[i, j])
                })

results_df = pd.DataFrame(results)
results_df.to_csv(RESULTS_FILE, index=False)

with open(REPORT_FILE, "w", encoding="utf-8") as file:
    file.write("\n".join(report_sections))

pd.DataFrame(matrix_rows).to_csv(MATRIX_FILE, index=False)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

print("\nSaved evaluation files:")
print(RESULTS_FILE)
print(REPORT_FILE)
print(MATRIX_FILE)
