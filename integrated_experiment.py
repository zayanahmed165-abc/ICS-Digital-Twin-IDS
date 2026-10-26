import os
import time
from datetime import datetime

import pandas as pd
import psutil


# =========================================
# PROJECT PATHS
# =========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

TWIN_FILE = os.path.join(DATA_DIR, "digital_twin_data.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "integrated_dataset.csv")


# =========================================
# READ DIGITAL TWIN DATA
# =========================================

twin_data = pd.read_csv(TWIN_FILE)

print("Digital Twin records:", len(twin_data))
print("Collecting synchronized OS telemetry...")


# =========================================
# OS TELEMETRY COLLECTION
# =========================================

telemetry_rows = []

# CPU percentage ko initialize karo
psutil.cpu_percent(interval=None)

# Windows system drive
system_drive = os.environ.get("SystemDrive", "C:") + os.sep

for index in range(len(twin_data)):

    # CPU sample ke liye short interval
    cpu_usage = psutil.cpu_percent(interval=0.1)

    # RAM usage
    ram_usage = psutil.virtual_memory().percent

    # Windows system drive ka disk usage
    disk_usage = psutil.disk_usage(system_drive).percent

    # Running processes count
    process_count = len(psutil.pids())

    # Collection timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

    telemetry_rows.append({
        "os_timestamp": timestamp,
        "cpu_usage": cpu_usage,
        "ram_usage": ram_usage,
        "disk_usage": disk_usage,
        "process_count": process_count
    })

    if (index + 1) % 100 == 0:
        print(f"Collected OS samples: {index + 1}/{len(twin_data)}")


# =========================================
# COMBINE ICS + OS DATA
# =========================================

os_data = pd.DataFrame(telemetry_rows)

# Row-wise combination:
# Har Digital Twin sample ke saath uska collected OS sample
combined = pd.concat(
    [
        twin_data.reset_index(drop=True),
        os_data.reset_index(drop=True)
    ],
    axis=1
)


# =========================================
# VALIDATION
# =========================================

print("\nMissing values:")
print(combined.isnull().sum())

print("\nLabel distribution:")
print(combined["label"].value_counts())

print("\nTotal combined records:", len(combined))

# Verify matching row counts
assert len(twin_data) == len(os_data)
assert len(combined) == len(twin_data)


# =========================================
# SAVE INTEGRATED DATASET
# =========================================

combined.to_csv(OUTPUT_FILE, index=False)

print("\nIntegrated dataset created successfully!")
print("Saved at:", OUTPUT_FILE)

print("\nIntegrated columns:")
print(combined.columns.tolist())