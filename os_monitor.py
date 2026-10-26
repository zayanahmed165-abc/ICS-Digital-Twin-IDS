import psutil
import csv
import os
import time
from datetime import datetime

# Project ka main folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data folder
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# CSV file
CSV_FILE = os.path.join(DATA_DIR, "os_telemetry.csv")


# CSV file create karo
with open(CSV_FILE, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "timestamp",
        "cpu_usage",
        "ram_usage",
        "disk_usage",
        "process_count"
    ])

    # 20 samples collect karo
    for i in range(20):

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cpu_usage = psutil.cpu_percent(interval=1)
        ram_usage = psutil.virtual_memory().percent
        disk_usage = psutil.disk_usage("/").percent
        process_count = len(psutil.pids())

        writer.writerow([
            timestamp,
            cpu_usage,
            ram_usage,
            disk_usage,
            process_count
        ])

        print(
            f"Sample {i + 1}: "
            f"CPU={cpu_usage}% | "
            f"RAM={ram_usage}% | "
            f"Disk={disk_usage}% | "
            f"Processes={process_count}"
        )

print("\nOS telemetry collection complete!")
print("CSV saved at:", CSV_FILE)