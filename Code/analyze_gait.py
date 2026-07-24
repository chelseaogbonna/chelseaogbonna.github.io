import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from scipy.signal import find_peaks

# ======================================================
# DATA LOCATION
# ======================================================
folder_path = os.path.expanduser(
    "~/Documents/Haptic-Diabetic-Foot-Sensors/data/IMU_trials"
)

# ======================================================
# FIGURE SAVE LOCATION
# ======================================================
fig_folder = os.path.expanduser(
    "~/Documents/Haptic-Diabetic-Foot-Sensors/figures"
)

os.makedirs(fig_folder, exist_ok=True)

# ======================================================
# CONDITION MAP
# ======================================================
condition_map = {
    1: ("Right", "Normal"),
    2: ("Right", "Lean Left"),
    3: ("Right", "Lean Forward"),
    4: ("Right", "Lean Right"),
    5: ("Right", "Lean Back"),
    6: ("Left", "Normal"),
    7: ("Left", "Lean Left"),
    8: ("Left", "Lean Forward"),
    9: ("Left", "Lean Right"),
    0: ("Left", "Lean Back")
}

results = []

# ======================================================
# LOAD CSV FILES
# ======================================================
files = sorted(glob.glob(os.path.join(folder_path, "trial_*.csv")))
print("Files detected:", len(files))

# ======================================================
# PROCESS EACH TRIAL
# ======================================================
for file in files:

    trial_num = int(os.path.basename(file).split("_")[1].split(".")[0])
    ending = trial_num % 10

    leg, condition = condition_map[ending]

    data = pd.read_csv(file)

    avg_pitch = data["pitch"].mean()
    avg_roll = data["roll"].mean()

    # Accelerometer magnitude
    acc_mag = np.sqrt(
        data["acc_x"]**2 +
        data["acc_y"]**2 +
        data["acc_z"]**2
    )

    # Gyroscope magnitude
    gyro_mag = np.sqrt(
        data["gyro_x"]**2 +
        data["gyro_y"]**2 +
        data["gyro_z"]**2
    )

    avg_acc = acc_mag.mean()
    avg_gyro = gyro_mag.mean()

    results.append({
        "trial": trial_num,
        "leg": leg,
        "condition": condition,
        "avg_pitch": avg_pitch,
        "avg_roll": avg_roll,
        "avg_acc": avg_acc,
        "avg_gyro": avg_gyro
    })

# ======================================================
# CREATE DATAFRAME
# ======================================================
df = pd.DataFrame(results)

print(df.head())

# ======================================================
# FIGURE 1 — Pitch vs Roll by Condition
# ======================================================
plt.figure()

for condition in df["condition"].unique():
    subset = df[df["condition"] == condition]

    plt.scatter(
        subset["avg_pitch"],
        subset["avg_roll"],
        label=condition
    )

plt.xlabel("Average Pitch")
plt.ylabel("Average Roll")
plt.title("Walking Orientation by Condition")

plt.legend()

plt.savefig(
    os.path.join(fig_folder, "imu_pitch_roll_by_condition.png"),
    dpi=300
)

plt.show()

# ======================================================
# FIGURE 2 — Left vs Right Leg
# ======================================================
plt.figure()

for leg in ["Left", "Right"]:

    subset = df[df["leg"] == leg]

    plt.scatter(
        subset["avg_pitch"],
        subset["avg_roll"],
        label=leg
    )

plt.xlabel("Average Pitch")
plt.ylabel("Average Roll")

plt.title("Left vs Right Leg Comparison")

plt.legend()

plt.savefig(
    os.path.join(fig_folder, "imu_left_vs_right.png"),
    dpi=300
)

plt.show()

# ======================================================
# FIGURE 3 — Example Gait Cycle
# ======================================================
file = files[0]

data = pd.read_csv(file)

time = data["time"]
acc_z = data["acc_z"]

plt.figure()

plt.plot(time, acc_z)

plt.xlabel("Time (s)")
plt.ylabel("Vertical Acceleration")

plt.title("Gait Cycle (Step Pattern)")

plt.savefig(
    os.path.join(fig_folder, "imu_gait_cycle.png"),
    dpi=300
)

plt.show()

# ======================================================
# FIGURE 4 — Step Detection
# ======================================================
peaks, _ = find_peaks(acc_z, height=12)

plt.figure()

plt.plot(time, acc_z)

plt.plot(
    time.iloc[peaks],
    acc_z.iloc[peaks],
    "ro"
)

plt.xlabel("Time")
plt.ylabel("Acceleration")

plt.title("Detected Steps")

plt.savefig(
    os.path.join(fig_folder, "imu_step_detection.png"),
    dpi=300
)

plt.show()

print("Estimated steps:", len(peaks))

# ======================================================
# FIGURE 5 — Pitch Distribution
# ======================================================
plt.figure()

df.boxplot(
    column="avg_pitch",
    by="condition"
)

plt.ylabel("Pitch (degrees)")

plt.title("Pitch Distribution by Walking Condition")

plt.suptitle("")

plt.savefig(
    os.path.join(fig_folder, "imu_pitch_distribution.png"),
    dpi=300
)

plt.show()

# ======================================================
# FIGURE 6 — Accelerometer vs Gyroscope
# ======================================================
plt.figure()

plt.scatter(
    df["avg_acc"],
    df["avg_gyro"]
)

plt.xlabel("Acceleration Magnitude")
plt.ylabel("Gyroscope Magnitude")

plt.title("Accelerometer vs Gyroscope Activity")

plt.savefig(
    os.path.join(fig_folder, "imu_acc_vs_gyro.png"),
    dpi=300
)

plt.show()

# ======================================================
# FIGURE 7 — Sensor Separation by Condition
# ======================================================
plt.figure()

for condition in df["condition"].unique():

    subset = df[df["condition"] == condition]

    plt.scatter(
        subset["avg_acc"],
        subset["avg_gyro"],
        label=condition
    )

plt.xlabel("Acceleration Magnitude")
plt.ylabel("Gyroscope Magnitude")

plt.title("Sensor Comparison by Walking Condition")

plt.legend()

plt.savefig(
    os.path.join(fig_folder, "imu_sensor_condition_comparison.png"),
    dpi=300
)

plt.show()

print("\nIMU analysis complete. Figures saved to:")
print(fig_folder)