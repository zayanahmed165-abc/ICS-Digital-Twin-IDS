import pandas as pd
import os

# Project ka main folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")

# Files read karo
digital_twin = pd.read_csv(
    os.path.join(DATA_DIR, "digital_twin_data.csv")
)

os_data = pd.read_csv(
    os.path.join(DATA_DIR, "os_telemetry.csv")
)

print("Digital Twin records:", len(digital_twin))
print("OS telemetry records:", len(os_data))

# OS data ko repeat karke Digital Twin ke size ke barabar karna
os_repeated = pd.concat(
    [os_data] * ((len(digital_twin) // len(os_data)) + 1),
    ignore_index=True
)

os_repeated = os_repeated.iloc[:len(digital_twin)]

# Dono datasets combine karo
combined = pd.concat(
    [
        digital_twin.reset_index(drop=True),
        os_repeated.reset_index(drop=True)
    ],
    axis=1
)

# Missing values check
print("\nMissing values:")
print(combined.isnull().sum())

# Duplicate rows check
print("\nDuplicate rows:", combined.duplicated().sum())

# Duplicates remove
combined = combined.drop_duplicates()

# Save final dataset
OUTPUT_FILE = os.path.join(DATA_DIR, "final_dataset.csv")

combined.to_csv(OUTPUT_FILE, index=False)

print("\nFinal dataset created successfully!")
print("Total records:", len(combined))
print("Saved at:", OUTPUT_FILE)

print("\nColumns:")
print(list(combined.columns))