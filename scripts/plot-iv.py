"""Plot I-V curve

Sweep the control voltage slowly, i.e, set the control voltage, wait, measure
voltage and current monitors, average values

"""

import argparse
import time
import numpy as np 
import matplotlib.pyplot as plt

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
parser.add_argument("-s", "--speed", type=int, help="Speed setting for sweep: 1-3", default=3)
parser.add_argument("-p", "--param", type=str, help="Parameter file", default="../params.json")
parser.add_argument("-j", "--njunc", type=int, help="Number of junctions", default=1)
parser.add_argument("--pump", dest="pump", action="store_true", help="LO pumping")
parser.add_argument("--no-pump", dest="pump", action="store_false", help="No LO pumping")
parser.set_defaults(pump=False)
parser.add_argument("--shot", dest="shot", action="store_true", help="Analyze shot noise")
parser.add_argument("--no-shot", dest="shot", action="store_false", help="Don't analyze shot noise")
parser.set_defaults(shot=True)
parser.add_argument("--vmax", type=float, help="Maximum control voltage", default=None)
parser.add_argument("--vmin", type=float, help="Minimum control voltage", default=None)
parser.add_argument("--current-offset", dest="i_offset", type=float, help="Current offset in uA", default=None)
parser.add_argument("--if-offset", dest="if_offset", type=float, help="IF power offset offset in uW", default=0)
parser.add_argument("--resistor", action="store_true", help="Measure simple resistance", default=False)
args = parser.parse_args()

# Sweep parameters
if args.vmax is None:
    vmax = 0.7 * args.njunc
else:
    vmax = args.vmax
if args.vmin is None:
    vmin = -vmax
else:
    vmin = args.vmin

# Parameters -----------------------------------------------------------------

if args.speed == 3:
    # Bias sweep: slowish
    average = 32
    npts = 101
    sleep_time = 0.05
    control_voltage = np.linspace(vmin, vmax, npts)
elif args.speed == 2:
    # Bias sweep: slowish
    average = 32
    npts = 501
    sleep_time = 0.2
    control_voltage = np.linspace(vmin, vmax, npts)
elif args.speed == 1:
    # Bias sweep: real slow
    average = 128
    npts = 1001
    sleep_time = 0.2
    control_voltage = np.linspace(vmin, vmax, npts)
else:
    raise ValueError

# Fit slope
vmin_slope = 5.5 * args.njunc  # mV
vmax_slope = 7.5 * args.njunc  # mV

# Sweep bias voltage ---------------------------------------------------------

# Initialize SIS bias control
bias = SISBias(param_file=args.param)

# Sweep bias voltage and measure voltage/current/IF power
voltage = np.empty_like(control_voltage)
current = np.empty_like(control_voltage)
ifpower = np.empty_like(control_voltage)
for i, _vctrl in np.ndenumerate(control_voltage):
    bias.set_control_voltage(_vctrl)
    time.sleep(sleep_time)
    vtmp, itmp, ptmp = np.empty(average), np.empty(average), np.empty(average)
    for j in range(average):
        vtmp[j] = bias.read_voltage()  # mV
        itmp[j] = bias.read_current()  # uA
        ptmp[j] = bias.read_ifpower()  # A.U.
    voltage[i] = np.mean(vtmp)
    current[i] = np.mean(itmp)
    ifpower[i] = np.mean(ptmp)
    print("{:4d}/{:4d}, {:6.3f} V, {:5.2f} mV, {:6.1f} uA, {:5.3f} uW".format(i[0], npts, _vctrl, voltage[i], current[i], ifpower[i]))

# Sort by voltage
idx = voltage.argsort()
voltage, current, ifpower = voltage[idx], current[idx], ifpower[idx]

# Correct for current offset
v_tmp = np.linspace(voltage.min()+0.5, voltage.max()-0.5, 101)
if args.i_offset is None:
    current_offset = np.mean(np.interp(v_tmp, voltage, current) - \
                             np.interp(v_tmp, -voltage[::-1], -current[::-1])) / 2
    print("\nCurrent offset: {:.2f} uA".format(current_offset))
    current -= current_offset
else:
    current -= args.i_offset

# Find intercept and normal resistance
if args.resistor:
    pnormal = np.polyfit(voltage, current, 1)
    rnormal = 1000 / pnormal[0]
    print("\nNormal resistance: {:.1f} ohms".format(rnormal))
else:
    mask = (vmin_slope <= voltage) & (voltage <= vmax_slope)
    pnormal = np.polyfit(voltage[mask], current[mask], 1)
    v_intercept = -pnormal[1] / pnormal[0]
    print("\nIntercept: {:.3f} mV".format(v_intercept))
    rnormal = 1000 / pnormal[0]
    print("\nNormal resistance: {:.1f} ohms".format(rnormal))

# Calibrate IF power
print("\nMean IF power: {:.4f} uW".format(np.mean(ifpower)))
ifpower -= args.if_offset  # offset
mask = (vmin_slope <= voltage) & (voltage <= vmax_slope)
pshot = np.polyfit(voltage[mask], ifpower[mask], 1)
ifpower_k = ifpower / pshot[0] * 5.8 / args.njunc  # TODO: check

# Calculate IF noise
mask = (vmin_slope <= voltage) & (voltage <= vmax_slope)
pifnoise = np.polyfit(voltage[mask], ifpower_k[mask], 1)
if_noise = np.polyval(pifnoise, v_intercept)
gamma = (50 - rnormal) / (50 + rnormal)
gmismatch = 1 - np.abs(gamma) ** 2
if_noise_corr = (if_noise - 1.3) * gmismatch
print("\nIF noise: {:.1f} K".format(if_noise))
print("\nIF noise: {:.1f} K (corrected)".format(if_noise_corr))
print("")

# Plot I-V curve
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
ax1.axhline(0, c='k', lw=0.5)
ax1.axvline(0, c='k', lw=0.5)
if vmin == -vmax:
    ax1.plot(-voltage, -current, 'k-', alpha=0.5)
if args.pump:
    ax1.plot(voltage, current, 'k-', label='I-V curve')
else:
    ax1.plot(voltage, current, 'k-', label='DC I-V curve')
    ax1.plot(voltage[voltage > 0], np.polyval(pnormal, voltage[voltage > 0]), 'r-', lw=0.5, label=r"$R_n={:.1f}~\Omega$".format(rnormal))
    ax1.plot(v_intercept, 0, 'r*', ms=10, label="$x$-intercept")
ax1.set_xlabel("Voltage (mV)")
ax1.set_ylabel("Current (uA)")
ax1.legend(loc=2, frameon=True)

# Plot IF power
ax2.axvline(0, c='k', lw=0.5)

if args.shot:
    ax2.plot(voltage, ifpower_k, 'k-', label='IF power')
    if vmin == -vmax:
        ax2.plot(-voltage, ifpower_k, 'k-', alpha=0.5)
    ax2.plot(voltage, np.polyval(pifnoise, voltage), 'r-', lw=0.5)
    tif_str = r"$T_\mathrm{IF}$"
    ax2.plot(v_intercept, if_noise, 'r*', ms=10, label="{}: {:.1f} K".format(tif_str, if_noise))
    tif_str = r"$T_\mathrm{IF}^\prime$"
    ax2.plot(v_intercept, if_noise_corr, 'b*', ms=10, label="{}: {:.1f} K".format(tif_str, if_noise_corr))
    ax2.set_ylabel("IF Power (K)")
else:
    ax2.plot(voltage, ifpower, 'k-', label='IF power')
    if vmin == -vmax:
        ax2.plot(-voltage, ifpower, 'k-', alpha=0.5)
    ax2.set_ylabel("IF Power (uW)")
ax2.set_xlabel("Voltage (mV)")
ax2.set_ylim(ymin=0)
ax2.legend(loc=4, frameon=True)

plt.show()
