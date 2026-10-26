import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# Project folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset path
DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "final_dataset.csv"
)

# Dataset load
data = pd.read_csv(DATA_FILE)

print("Dataset loaded!")
print("Records:", len(data))

# Features
features = [
    "temperature",
    "pressure",
    "flow_rate",
    "tank_level",
    "cpu_usage",
    "ram_usage",
    "disk_usage",
    "process_count"
]

X = data[features]

# Target
# Normal = 0
# Anomaly = 1
y = data["label"].apply(
    lambda x: 0 if x == "Normal" else 1
)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nRandom Forest Results")
print("---------------------")
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =========================
# Model 2: SVM
# =========================

from sklearn.svm import SVC

svm_model = SVC()

# Train SVM
svm_model.fit(X_train, y_train)

# Prediction
y_pred_svm = svm_model.predict(X_test)

# Results
print("\nSVM Results")
print("---------------------")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_svm))


# =========================
# Model 3: Isolation Forest
# =========================

# Train Isolation Forest
from sklearn.ensemble import IsolationForest
iso_model = IsolationForest(
    contamination=0.46,
    random_state=42
)

iso_model.fit(X_train)

# Prediction
y_pred_iso = iso_model.predict(X_test)

# Isolation Forest:
#  1  = Normal
# -1  = Anomaly
#
# Hamare dataset mein:
#  0 = Normal
#  1 = Attack
#
# Isliye convert kar rahe hain:
#  1  -> 0 (Normal)
# -1  -> 1 (Attack)

y_pred_iso = [0 if prediction == 1 else 1 for prediction in y_pred_iso]

# Results
print("\nIsolation Forest Results")
print("---------------------")
print("Accuracy:", accuracy_score(y_test, y_pred_iso))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_iso))

# =========================
# Confusion Matrix + FP/FN
# =========================

from sklearn.metrics import confusion_matrix

# Random Forest
cm_rf = confusion_matrix(y_test, y_pred)
tn_rf, fp_rf, fn_rf, tp_rf = cm_rf.ravel()

print("\nRandom Forest Confusion Matrix")
print(cm_rf)
print("False Positives:", fp_rf)
print("False Negatives:", fn_rf)

# SVM
cm_svm = confusion_matrix(y_test, y_pred_svm)
tn_svm, fp_svm, fn_svm, tp_svm = cm_svm.ravel()

print("\nSVM Confusion Matrix")
print(cm_svm)
print("False Positives:", fp_svm)
print("False Negatives:", fn_svm)

# Isolation Forest
cm_iso = confusion_matrix(y_test, y_pred_iso)
tn_iso, fp_iso, fn_iso, tp_iso = cm_iso.ravel()

print("\nIsolation Forest Confusion Matrix")
print(cm_iso)
print("False Positives:", fp_iso)
print("False Negatives:", fn_iso)

import numpy as np

print("\n=========================")
print("Robustness Evaluation")
print("=========================")

# Add small random noise to test data
np.random.seed(42)

X_test_noisy = X_test.copy()

noise = np.random.normal(
    loc=0,
    scale=0.05,
    size=X_test_noisy.shape
)

X_test_noisy = X_test_noisy + noise


# -------- Random Forest --------
# Your Random Forest variable is: model
rf_noisy_pred = model.predict(X_test_noisy)

rf_noisy_accuracy = accuracy_score(
    y_test,
    rf_noisy_pred
)

print("\nRandom Forest with Noise")
print("-------------------------")
print("Accuracy:", rf_noisy_accuracy)


# -------- SVM --------
svm_noisy_pred = svm_model.predict(X_test_noisy)

svm_noisy_accuracy = accuracy_score(
    y_test,
    svm_noisy_pred
)

print("\nSVM with Noise")
print("-------------------------")
print("Accuracy:", svm_noisy_accuracy)


# -------- Isolation Forest --------
iso_noisy_pred = iso_model.predict(X_test_noisy)

# Isolation Forest:
# 1  = Normal
# -1 = Anomaly

iso_noisy_pred = [
    0 if prediction == 1 else 1
    for prediction in iso_noisy_pred
]

iso_noisy_accuracy = accuracy_score(
    y_test,
    iso_noisy_pred
)

print("\nIsolation Forest with Noise")
print("-------------------------")
print("Accuracy:", iso_noisy_accuracy)

# =========================
# Ablation Study
# =========================

print("\n=========================")
print("Ablation Study")
print("=========================")

# Remove one feature: cpu_usage
ablation_features = [
    "temperature",
    "pressure",
    "flow_rate",
    "tank_level",
    "ram_usage",
    "disk_usage",
    "process_count"
]

X_ablation = data[ablation_features]

# Same train/test split
X_train_ab, X_test_ab, y_train_ab, y_test_ab = train_test_split(
    X_ablation,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Train Random Forest without cpu_usage
ablation_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

ablation_model.fit(X_train_ab, y_train_ab)

# Prediction
ablation_pred = ablation_model.predict(X_test_ab)

# Accuracy
ablation_accuracy = accuracy_score(
    y_test_ab,
    ablation_pred
)

print("\nRandom Forest without cpu_usage")
print("-------------------------")
print("Accuracy:", ablation_accuracy)

print("\nOriginal Random Forest Accuracy:", accuracy)
print("Accuracy Change:", accuracy - ablation_accuracy)
