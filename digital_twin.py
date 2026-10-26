import random
import csv
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

CSV_FILE = os.path.join(DATA_DIR, "digital_twin_data.csv")


def generate_data():
    rows = []

    # -------------------------
    # 1. NORMAL DATA
    # -------------------------
    for i in range(1000):

        temperature = random.uniform(20, 30)
        pressure = random.uniform(40, 60)
        flow_rate = random.uniform(80, 100)
        tank_level = random.uniform(40, 80)

        rows.append([
            i,
            temperature,
            pressure,
            flow_rate,
            tank_level,
            "Normal"
        ])

    # -------------------------
    # 2. TEMPERATURE ANOMALY
    # -------------------------
    for i in range(1000, 1100):

        temperature = random.uniform(50, 70)
        pressure = random.uniform(40, 60)
        flow_rate = random.uniform(80, 100)
        tank_level = random.uniform(40, 80)

        rows.append([
            i,
            temperature,
            pressure,
            flow_rate,
            tank_level,
            "Temperature_Anomaly"
        ])

    # -------------------------
    # 3. PRESSURE ANOMALY
    # -------------------------
    for i in range(1100, 1200):

        temperature = random.uniform(20, 30)
        pressure = random.uniform(80, 100)
        flow_rate = random.uniform(80, 100)
        tank_level = random.uniform(40, 80)

        rows.append([
            i,
            temperature,
            pressure,
            flow_rate,
            tank_level,
            "Pressure_Anomaly"
        ])

    # -------------------------
    # 4. SENSOR ANOMALY
    # -------------------------
    for i in range(1200, 1300):

        temperature = random.uniform(20, 30)
        pressure = random.uniform(40, 60)
        flow_rate = random.uniform(150, 200)
        tank_level = random.uniform(5, 15)

        rows.append([
            i,
            temperature,
            pressure,
            flow_rate,
            tank_level,
            "Sensor_Anomaly"
        ])

    return rows


# Generate dataset
data = generate_data()


# Save dataset
with open(CSV_FILE, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "time",
        "temperature",
        "pressure",
        "flow_rate",
        "tank_level",
        "label"
    ])

    writer.writerows(data)


print("Digital Twin dataset generated successfully!")
print("Total records:", len(data))
print("CSV saved at:", CSV_FILE)