import time
import numpy as np 
import matplotlib.pyplot as plt
from qmix.mathfn.filters import gauss_conv

import sys
sys.path.append("..")
from sisbias import SISBias

# Sweep parameters
period = 0.2
npts = 10_000 * period
vmin = -0.7
vmax = 0.7

# Start I-V bias sweeps
bias = SISBias(param_file="../params.json")
bias.sweep_control_voltage(vmin, vmax, npts=npts, sweep_period=period)
bias.start_iv_monitor_scan(npts=npts, sweep_period=period)

# Create figure
plt.ion()
fig, ax1 = plt.subplots(figsize=(6,5))
ax1.set_xlabel("Voltage (mV)")
ax1.set_ylabel("Current (uA)")
ax1.set_xlim([-5.5, 5.5])
ax1.set_ylim([-450, 450])
line1, = ax1.plot([0], [0], 'k.', ms=1)
fig.canvas.draw()
plt.show()

while True:
    try:
        # Read I-V curve
        voltage, current, _ = bias.read_iv_curve()
        
        # Draw I-V curve
        line1.set_data(voltage*1e3, current*1e6)
        fig.canvas.draw()
        plt.pause(0.0001)
        fig.canvas.flush_events()

        # Restart scans
        bias.sweep_control_voltage(vmin, vmax, npts=npts, sweep_period=period, verbose=False)
        bias.start_iv_monitor_scan(npts=npts, sweep_period=period, verbose=False)
        time.sleep(period)

    except KeyboardInterrupt:
        plt.close('all')
        break
