"""Initialize parameters file for SIS bias control."""

import json

from appdirs import user_config_dir
import argparse

# Grab arguments
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--configfile", type=str, help="Config file", default=None)
args = parser.parse_args()


# Default parameters
params = dict(
    # Control voltage
    VCTRL = dict(
        AO_N_CHANNEL = 0,
        AO_P_CHANNEL = 1,
        ),
    # Voltage monitor
    VMON = dict(
        AI_CHANNEL = 0,
        GAIN = -100, 
        OFFSET = 0,
        ),
    # Current monitor
    IMON = dict(
        AI_CHANNEL = 1,
        GAIN = -1500, 
        OFFSET = 0,
        ),
    # IF power
    PIF = dict(
        AI_CHANNEL = 2,
        OFFSET = 0,
        ),
)


# Location of config file
if args.configfile is None:
    filename = user_config_dir("rxlab-sis-bias")
else:
    filename = args.configfile

# Save config file
with open(filename, 'w') as fout:
    json.dump(params, fout, indent=4)

print(f"Configuration file saved to: {filename}")
