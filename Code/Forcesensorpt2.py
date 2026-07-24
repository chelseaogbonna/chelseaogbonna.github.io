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

files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
print("Files detected:", len(files))

if len(files) == 0:
    raise ValueError(f"No CSV files found in {folder_path}")

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
# PROCESS EACH TRIAL
# ======================================================
for file in files:

    trial_num = int(os.path.basename(file).split("_")[1].split(".")[0])
    ending = trial_num % 10

    leg, condition = condition_map[ending]

    data = pd.read_csv(file)
    data.columns = data.columns.str.strip()

    time = data["time"]
    force = data["force_n"]

    # --------------------------------------------------
    # STEP DETECTION
    # --------------------------------------------------
    threshold = force.mean() + force.std()

    peaks, _ = find_peaks(
        force,
        height=threshold,
        distance=10
    )

    num_steps = len(peaks)

    # --------------------------------------------------
    # CADENCE
    # --------------------------------------------------
    duration = time.iloc[-1] - time.iloc[0]

    cadence = num_steps / duration if duration > 0 else np.nan

    # --------------------------------------------------
    # STRIDE VARIABILITY
    # --------------------------------------------------
    if len(peaks) > 1:
        step_intervals = np.diff(time.iloc[peaks].values)
        stride_variability = np.std(step_intervals)
    else:
        stride_variability = np.nan

    # --------------------------------------------------
    # PEAK FORCE METRICS
    # --------------------------------------------------
    if len(peaks) > 0:
        peak_forces = force.iloc[peaks].values

        avg_peak_force = np.mean(peak_forces)
        std_peak_force = np.std(peak_forces)

    else:
        avg_peak_force = np.nan
        std_peak_force = np.nan

    results.append({
        "trial": trial_num,
        "leg": leg,
        "condition": condition,
        "avg_force": force.mean(),
        "max_force": force.max(),
        "std_force": force.std(),
        "num_steps": num_steps,
        "cadence": cadence,
        "stride_variability": stride_variability,
        "avg_peak_force": avg_peak_force,
        "std_peak_force": std_peak_force
    })

# ======================================================
# CREATE DATAFRAME
# ======================================================
df = pd.DataFrame(results)

# Save processed data
output_csv = os.path.expanduser(
    "~/Documents/force_gait_results_by_condition.csv"
)

df.to_csv(output_csv, index=False)

print("\nResults saved to:", output_csv)

# ======================================================
# VISUALIZATION: FORCE FEATURES BY CONDITION
# ======================================================
features = [
    "avg_force",
    "avg_peak_force",
    "std_peak_force",
    "stride_variability",
    "cadence"
]

feat_labels = [
    "Average Force",
    "Average Peak Force",
    "Peak Force Variability",
    "Stride Variability",
    "Cadence"
]

# ------------------------------------------------------
# BOX PLOTS FOR EACH FEATURE
# ------------------------------------------------------
for f, label in zip(features, feat_labels):

    plt.figure(figsize=(10,5))

    df.boxplot(
        column=f,
        by="condition"
    )

    plt.ylabel(label)

    plt.title(f"{label} by Walking Condition")

    plt.suptitle("")

    plt.xticks(rotation=15)

    plt.tight_layout()

    # Save figure
    filename = f"force_{f}_by_condition.png"

    plt.savefig(
        os.path.join(fig_folder, filename),
        dpi=300
    )

    plt.show()

# ======================================================
# SUMMARY TABLE
# ======================================================
summary = df.groupby("condition")[features].mean().round(2)

summary.columns = feat_labels

print("\n===== FORCE GAIT SUMMARY TABLE BY CONDITION =====")
print(summary.to_string())

print("\nHigher stride variability, peak force variability, or lower cadence can indicate less stable walking.")

print("\nFigures saved to:", fig_folder)