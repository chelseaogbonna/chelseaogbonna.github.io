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
    "~/Documents/Haptic-Diabetic-Foot-Sensors/data/force_trials"
)

# ======================================================
# FIGURE SAVE LOCATION
# ======================================================
fig_folder = os.path.expanduser(
    "~/Documents/Haptic-Diabetic-Foot-Sensors/figures"
)

os.makedirs(fig_folder, exist_ok=True)

files = sorted(glob.glob(os.path.join(folder_path, "trial_*.csv")))

print("Files detected:", len(files))

if len(files) == 0:
    raise ValueError("No CSV files found")

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
# PROCESS FILES
# ======================================================
for file in files:

    trial_num = int(os.path.basename(file).split("_")[1].split(".")[0])
    ending = trial_num % 10

    leg, condition = condition_map[ending]

    data = pd.read_csv(file)
    data.columns = data.columns.str.strip()

    time = data["time"]
    force = data["force_n"]

    # Step detection
    threshold = force.mean() + force.std()

    peaks, _ = find_peaks(force, height=threshold, distance=10)

    num_steps = len(peaks)

    # Cadence
    duration = time.iloc[-1] - time.iloc[0]
    cadence = num_steps / duration if duration > 0 else np.nan

    # Stride variability
    if len(peaks) > 1:
        step_intervals = np.diff(time.iloc[peaks].values)
        stride_variability = np.std(step_intervals)
    else:
        stride_variability = np.nan

    # Peak force metrics
    if len(peaks) > 0:
        peak_forces = force.iloc[peaks]
        avg_peak_force = peak_forces.mean()
        std_peak_force = peak_forces.std()
    else:
        avg_peak_force = np.nan
        std_peak_force = np.nan

    results.append({
        "trial": trial_num,
        "condition": condition,
        "avg_force": force.mean(),
        "avg_peak_force": avg_peak_force,
        "std_peak_force": std_peak_force,
        "stride_variability": stride_variability,
        "cadence": cadence
    })

# ======================================================
# CREATE DATAFRAME
# ======================================================
df = pd.DataFrame(results)

# ======================================================
# CREATE NORMAL vs ABNORMAL LABEL
# ======================================================
df["gait_type"] = df["condition"].apply(
    lambda x: "Normal" if x == "Normal" else "Abnormal"
)

print(df.head())

# ======================================================
# FEATURES
# ======================================================
features = [
    "avg_force",
    "avg_peak_force",
    "std_peak_force",
    "stride_variability",
    "cadence"
]

labels = [
    "Average Force",
    "Average Peak Force",
    "Peak Force Variability",
    "Stride Variability",
    "Cadence"
]

# ======================================================
# PLOTS (NORMAL vs ABNORMAL)
# ======================================================
for feature, label in zip(features, labels):

    plt.figure(figsize=(7,5))

    df.boxplot(column=feature, by="gait_type")

    plt.title(f"{label}: Normal vs Abnormal Walking")

    plt.ylabel(label)

    plt.suptitle("")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            fig_folder,
            f"{feature}_normal_vs_abnormal.png"
        ),
        dpi=300
    )

    plt.show()

# ======================================================
# SUMMARY TABLE
# ======================================================
summary = df.groupby("gait_type")[features].mean().round(2)

print("\n===== NORMAL vs ABNORMAL SUMMARY =====\n")

print(summary)