import time
import numpy as np
import matplotlib.pyplot as plt
from nlabapi import LabBench

# Parameters
number_of_samples = 10  # Samples per read
sample_rate = 8000.0  # Hz
duration = 10  # Duration in seconds

def collect_data(nlab, sample_rate, number_of_samples, duration):
    data_points = []
    timestamps = []
    start_time = time.time()
    
    while time.time() - start_time < duration:
        data = nlab.read_all_channels(sample_rate, number_of_samples)
        avg_voltage = sum(data[0]) / len(data[0])  # Average voltage from Ch1
        data_points.append(avg_voltage)
        timestamps.append(time.time() - start_time)
    
    return np.array(timestamps), np.array(data_points)

# Connect to nLab
nlab = LabBench.open_first_available()

# Collect data
times, voltages = collect_data(nlab, sample_rate, number_of_samples, duration)

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(times, voltages, label='Voltage Signal', color='b')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.title('Voltage Signal Over Time')
plt.legend()
plt.grid()
plt.show()
