import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from scipy.signal import find_peaks

# ================================================================
# SECTION 1: SETUP & DATA LOADING
# ================================================================

folder_path = os.path.expanduser(
    "~/Documents/Haptic-Diabetic-Foot-Sensors/data/IMU_trials"
)

# Folder where figures will be saved
fig_folder = os.path.expanduser(
    "~/Documents/Haptic-Diabetic-Foot-Sensors/figures"
)

os.makedirs(fig_folder, exist_ok=True)

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

files = sorted(glob.glob(os.path.join(folder_path, "trial_*.csv")))
print(f"Files detected: {len(files)}")

# ================================================================
# SECTION 2: FEATURE EXTRACTION
# ================================================================

for file in files:

    trial_num = int(os.path.basename(file).split("_")[1].split(".")[0])
    ending = trial_num % 10
    leg, condition = condition_map[ending]

    data = pd.read_csv(file)

    avg_pitch = data["pitch"].mean()
    avg_roll  = data["roll"].mean()

    std_pitch = data["pitch"].std()
    std_roll  = data["roll"].std()

    acc_mag = np.sqrt(data["acc_x"]**2 + data["acc_y"]**2 + data["acc_z"]**2)
    gyro_mag = np.sqrt(data["gyro_x"]**2 + data["gyro_y"]**2 + data["gyro_z"]**2)

    avg_acc  = acc_mag.mean()
    avg_gyro = gyro_mag.mean()

    std_acc  = acc_mag.std()
    std_gyro = gyro_mag.std()

# ================================================================
# SECTION 3: STEP DETECTION
# ================================================================

    acc_z = data["acc_z"]
    time  = data["time"]

    peaks, _ = find_peaks(
        acc_z,
        height=acc_z.mean() + acc_z.std(),
        distance=5
    )

    num_steps = len(peaks)

    trial_duration = time.iloc[-1] - time.iloc[0]
    cadence = num_steps / trial_duration if trial_duration > 0 else 0

    if len(peaks) > 1:
        step_intervals = np.diff(time.iloc[peaks].values)
        stride_regularity = np.std(step_intervals)
    else:
        stride_regularity = np.nan

# ================================================================
# SECTION 4: PEAK HEIGHT ANALYSIS
# ================================================================

    if len(peaks) > 0:
        peak_heights = acc_z.iloc[peaks].values
        avg_peak_height = np.mean(peak_heights)
        std_peak_height = np.std(peak_heights)
    else:
        avg_peak_height = np.nan
        std_peak_height = np.nan

    results.append({
        "trial": trial_num,
        "leg": leg,
        "condition": condition,
        "avg_pitch": avg_pitch,
        "avg_roll": avg_roll,
        "std_pitch": std_pitch,
        "std_roll": std_roll,
        "avg_acc": avg_acc,
        "avg_gyro": avg_gyro,
        "std_acc": std_acc,
        "std_gyro": std_gyro,
        "num_steps": num_steps,
        "cadence": cadence,
        "stride_regularity": stride_regularity,
        "avg_peak_height": avg_peak_height,
        "std_peak_height": std_peak_height
    })

df = pd.DataFrame(results)

print("\nSample extracted features:")
print(df.head())

# ================================================================
# SECTION 5: VISUALIZATION
# ================================================================

# Pitch variability
plt.figure(figsize=(10,5))
df.boxplot(column="std_pitch", by="condition")

plt.ylabel("Std Dev of Pitch (degrees)")
plt.title("Pitch Variability by Walking Condition")
plt.suptitle("")
plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(os.path.join(fig_folder, "imu_pitch_variability.png"), dpi=300)

plt.show()

# Gyroscope variability
plt.figure(figsize=(10,5))
df.boxplot(column="std_gyro", by="condition")

plt.ylabel("Std Dev of Gyroscope Magnitude")
plt.title("Gyroscope Variability by Walking Condition")
plt.suptitle("")
plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(os.path.join(fig_folder, "imu_gyro_variability.png"), dpi=300)

plt.show()

# Cadence
plt.figure(figsize=(10,5))
df.boxplot(column="cadence", by="condition")

plt.ylabel("Steps per Second")
plt.title("Cadence by Walking Condition")
plt.suptitle("")
plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(os.path.join(fig_folder, "imu_cadence.png"), dpi=300)

plt.show()

# Stride regularity
plt.figure(figsize=(10,5))
df.boxplot(column="stride_regularity", by="condition")

plt.ylabel("Stride Interval Std Dev (s)")
plt.title("Stride Regularity by Walking Condition")
plt.suptitle("")
plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(os.path.join(fig_folder, "imu_stride_regularity.png"), dpi=300)

plt.show()

# Peak height variability
plt.figure(figsize=(10,5))
df.boxplot(column="std_peak_height", by="condition")

plt.ylabel("Std Dev of Step Impact")
plt.title("Footfall Variability by Walking Condition")
plt.suptitle("")
plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(os.path.join(fig_folder, "imu_peak_variability.png"), dpi=300)

plt.show()

# ================================================================
# SECTION 6: FEATURE COMPARISON
# ================================================================

plt.figure(figsize=(12,6))

features = ["std_pitch", "std_gyro", "stride_regularity", "std_peak_height"]
labels = ["Pitch\nVariability", "Gyro\nVariability", "Stride\nRegularity", "Footfall\nVariability"]

normal_means = [df[df["condition"] == "Normal"][f].mean() for f in features]
incorrect_means = [df[df["condition"] != "Normal"][f].mean() for f in features]

for i in range(len(features)):
    max_val = max(normal_means[i], incorrect_means[i])
    if max_val > 0:
        normal_means[i] /= max_val
        incorrect_means[i] /= max_val

x = np.arange(len(features))
width = 0.35

plt.bar(x - width/2, normal_means, width, label="Normal")
plt.bar(x + width/2, incorrect_means, width, label="Incorrect")

plt.xticks(x, labels)
plt.ylabel("Normalized Feature Value")

plt.title("Feature Comparison: Normal vs Incorrect Walking")

plt.legend()

plt.tight_layout()

plt.savefig(os.path.join(fig_folder, "imu_feature_comparison.png"), dpi=300)

plt.show()

# ================================================================
# SECTION 7: SUMMARY TABLE
# ================================================================

print("\n===== SUMMARY TABLE =====")

summary = df.groupby("condition")[
    ["std_pitch", "std_gyro", "cadence", "stride_regularity", "std_peak_height"]
].mean().round(3)

summary.columns = [
    "Pitch Variability",
    "Gyro Variability",
    "Cadence",
    "Stride Regularity",
    "Footfall Variability"
]

print(summary.to_string())

print("\nHigher variability values generally indicate more detectable incorrect walking.")

print("\nFigures saved to:", fig_folder)