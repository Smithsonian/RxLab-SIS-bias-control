"""Plot I-V curve

Sweep the control voltage fast. Allows for real-time tuning.

"""

import argparse
import time
import numpy as np 
import matplotlib.pyplot as plt
from qmix.mathfn.filters import gauss_conv

import sys
sys.path.append("..")
from sisbias import SISBias

# Matplotlib formatting, optional
try:
    plt.style.use(["science", "sans", "no-latex"])
except:
    print("Matplotlib styles not found")
    print("\ttry: pip install SciencePlots")

# Arguments ------------------------------------------------------------------

# Grab arguments
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--fsample", type=int, help="Sampling frequency (samples per second)", default=10_000)
parser.add_argument("-p", "--period", type=float, help="Sweep period", default=0.2)
parser.add_argument("--vmax", type=float, help="Maximum control voltage", default=None)
parser.add_argument("--vmin", type=float, help="Minimum control voltage", default=None)
parser.add_argument("--plot-if", dest="plot_if", action="store_true", help="Plot IF power")
parser.add_argument("--control-off", dest="vctrl_off", action="store_true", help="Stop control voltage sweep")
args = parser.parse_args()

# Sweep parameters
period = args.period
npts = args.fsample * period
if args.vmax is None:
    vmax = 1
else:
    vmax = args.vmax
if args.vmin is None:
    vmin = -vmax
else:
    vmin = args.vmin

# ----------------------------------------------------------------------------

# Start I-V bias sweeps
bias = SISBias(param_file="../params.json")
if not args.vctrl_off:
    bias.sweep_control_voltage(vmin, vmax, npts=npts, sweep_period=period)
bias.start_iv_monitor_scan(npts=npts, sweep_period=period)
time.sleep(period * 2)

# Read I-V curve
voltage, current, ifpower = bias.read_iv_curve()

# Create figure
plt.ion()
if args.plot_if:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
    ax2.set_xlim([voltage.min() * 1e3, voltage.max() * 1e3])
    ax2.set_xlabel("Voltage (mV)")
    ax2.set_ylim([0, 1])
    ax2.set_ylabel("IF power (uW)")
else:
    fig, ax1 = plt.subplots(figsize=(10,8))
ax1.set_xlabel("Voltage (mV)")
ax1.set_ylabel("Current (uA)")
ax1.set_xlim([voltage.min() * 1e3, voltage.max() * 1e3])
ax1.set_ylim([current.min() * 1e6, current.max() * 1e6])
if args.plot_if:
    line1, = ax1.plot([0], [0], 'k.', ms=1)
    line2, = ax2.plot([0], [0], 'k.', ms=1)
else:
    line1, = ax1.plot([0], [0], 'k.', ms=1)
fig.canvas.draw()
plt.show()

while True:
    try:
        # Restart scans
        if not args.vctrl_off:
            bias.sweep_control_voltage(vmin, vmax, npts=npts, sweep_period=period, verbose=False)
        bias.start_iv_monitor_scan(npts=npts, sweep_period=period, verbose=False)
        time.sleep(period)

        # Read I-V curve
        voltage, current, ifpower = bias.read_iv_curve()
        
        # Draw I-V curve
        line1.set_data(voltage*1e3, current*1e6)
        if args.plot_if:
            line2.set_data(voltage*1e3, ifpower)
        fig.canvas.draw()
        plt.pause(0.0001)
        fig.canvas.flush_events()

    except KeyboardInterrupt:
        plt.close('all')
        break