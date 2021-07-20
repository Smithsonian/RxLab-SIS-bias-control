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
bias = SISBias(param_file="../params.json")

# Set constant bias voltage
bias.set_control_voltage(0.61)
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
t = np.linspace(0, total_time, npts)

# Print statistics
print("Voltage:")
print("\tMean:               {:5.1f} mV".format(np.mean(voltage)))
print("\tStandard deviation: {:5.1f} uV".format(np.std(voltage)*1000))
print("\t                    {:5.1f} % ".format(np.std(voltage)/np.mean(voltage)*100))
print("Current:")
print("\tMean:               {:5.1f} uA".format(np.mean(current)))
print("\tStandard deviation: {:5.1f} uA".format(np.std(current)))
print("\t                    {:5.1f} % ".format(np.std(current)/np.mean(current)*100))
print("\nSampling frequency: {:.1f} kHz".format(npts*2/total_time/1e3))

# Plot
fig, ax = plt.subplots()
ax.plot(voltage, current, 'ko', ms=1, alpha=0.5)
p = np.polyfit(voltage, current, 1)
x = np.linspace(voltage.min(), voltage.max(), 3)
ax.plot(x, np.polyval(p, x), 'r-')
ax.set_xlabel("Voltage (mV)")
ax.set_ylabel("Current (uA)")

# Zero
voltage -= np.mean(voltage)
current -= np.mean(current)

# FFT
voltage_fft = np.fft.fftshift(np.fft.fft(voltage))
current_fft = np.fft.fftshift(np.fft.fft(current))
f = np.fft.fftshift(np.fft.fftfreq(len(voltage), d=t[1]-t[0]))

# # Mask for frequency
# f_idx = np.abs(f - 5).argmin()

# Plot
fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
ax1.plot(t*1e3, voltage*1e3, 'k', lw=0.5, alpha=0.2)
ax1.plot(t*1e3, gauss_conv(voltage, 3)*1e3, 'r')
ax1.axhspan(-np.std(voltage)*1e3, np.std(voltage)*1e3, color='r', alpha=0.2)
ax2.plot(f, np.abs(voltage_fft), 'k', lw=0.5, alpha=0.5)
ax2.axvspan(55, 65, color='r', alpha=0.2)
ax3.plot(t*1e3, current*1e3, 'k', lw=0.5, alpha=0.2)
ax3.plot(t*1e3, gauss_conv(current, 3)*1e3, 'r')
ax3.axhspan(-np.std(current)*1e3, np.std(current)*1e3, color='r', alpha=0.2)
ax4.plot(f, np.abs(current_fft), 'k', lw=0.5, alpha=0.5)
ax4.axvspan(55, 65, color='r', alpha=0.2)
ax1.set_xlabel("Time (ms)")
ax2.set_xlabel("Frequency (Hz)")
ax3.set_xlabel("Time (ms)")
ax4.set_xlabel("Frequency (Hz)")
ax1.set_ylabel("Voltage (mV")
ax2.set_ylabel("Voltage")
ax3.set_ylabel("Current")
ax4.set_ylabel("Current")
ax1.set_xlim([0, t.max()*1e3])
ax2.set_xlim([0, 500])
ax2.set_ylim(ymin=0)
ax3.set_xlim([0, t.max()*1e3])
ax4.set_xlim([0, 500])
ax4.set_ylim(ymin=0)

plt.show()

# Close connection to DAQ device
bias.close()
