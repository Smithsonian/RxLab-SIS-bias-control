"""Set constant control voltage and then measure the voltage and current
monitors.

"""

import time
import numpy as np 
import matplotlib.pyplot as plt
from qmix.mathfn.filters import gauss_conv

import sys
sys.path.append("..")
from sisbias import SISBias

# Initialize DAQ device
bias = SISBias()

# Set constant bias voltage
bias.set_control_voltage(-0.5)
time.sleep(0.1)

# Sample voltage/current monitors
npts = 1001
voltage = np.empty(npts)
current = np.empty(npts)
start = time.time()
for i in range(npts):
    voltage[i] = bias.read_voltage()
    current[i] = bias.read_current()
total_time = time.time() - start

# Print statistics
print("Voltage:")
print("\tMean:               {:5.1f} mV".format(np.mean(voltage)))
print("\tStandard deviation: {:5.1f} uV".format(np.std(voltage)*1000))
print("Current:")
print("\tMean:               {:5.1f} uA".format(np.mean(current)))
print("\tStandard deviation: {:5.1f} uA".format(np.std(current)))
print("\nSampling frequency: {:.1f} kHz".format(npts*2/total_time/1e3))

# Plot
fig, ax = plt.subplots()
ax.plot(voltage, current, 'ko', ms=1, alpha=0.5)
p = np.polyfit(voltage, current, 1)
x = np.linspace(voltage.min(), voltage.max(), 3)
ax.plot(x, np.polyval(p, x), 'r-')
ax.set_xlabel("Voltage (mV)")
ax.set_ylabel("Current (uA)")
plt.show()

# Zero
voltage -= np.mean(voltage)
current -= np.mean(current)

# Normalize
voltage /= np.abs(voltage).max()
current /= np.abs(current).max()

# Plot
fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 8))
t = np.linspace(0, total_time, npts)
ax1.plot(t*1e3, voltage, 'k', lw=0.5)
ax1.plot(t*1e3, gauss_conv(voltage, 3), 'r')
ax2.plot(t*1e3, current, 'k', lw=0.5)
ax2.plot(t*1e3, gauss_conv(current, 3), 'r')
ax1.set_xlabel("Time (ms)")
ax2.set_xlabel("Time (ms)")
ax1.set_ylabel("Voltage (normalized)")
ax2.set_ylabel("Current (normalized)")
plt.show()

# Close connection to DAQ device
bias.close()
