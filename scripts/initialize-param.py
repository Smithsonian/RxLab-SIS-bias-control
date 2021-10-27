"""Initialize parameters file for bin/sisbias"""

import json
from appdirs import user_config_dir
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--paramfile", type=str, help="Param file", default=None)
args = parser.parse_args()

# Default parameters
params = dict(
    # General info
    DEVICE="X",   # device number
    NJUNC=1,      # number of series SIS junctions
    # Voltage sweep
    VMIN=-1,      # control voltage sweep, minimum value [V]
    VMAX=1,       # control voltage sweep, maximum value [V]
    PERIOD=0.2,   # control voltage sweep, period [s]
    NPTS=1000,    # control voltage sweep, number of points
    VLIMIT=5,     # absolute maximum control voltage
    # Offsets / corrections
    IOFFSET=0,    # SIS current offset [uA]
    IFOFFSET=0,   # IF power offset [?W]
    IFCORR=1,     # IF conversion [K/?W]
    # Misc
    FREQ=0,       # LO frequency [GHz]
    LNA='on',     # Cryogenic LNA status
    IFFREQ=7.5,   # IF frequency [GHz]
    ICOIL=0,      # Coil current [mA]
)

# Location of param file
if args.paramfile is None:
    filename = user_config_dir("rxlab-sis-bias.param")
else:
    filename = args.configfile

# Save config file
with open(filename, 'w') as fout:
    json.dump(params, fout, indent=4)
print(f"\nParameter file saved to: {filename}\n")
