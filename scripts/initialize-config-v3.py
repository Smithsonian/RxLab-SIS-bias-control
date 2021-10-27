"""Initialize configuration file for SIS bias control."""

import json
from appdirs import user_config_dir
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--configfile", type=str, help="Config file", default=None)
args = parser.parse_args()

# Default configuration values (Ryan's bias board)
config = dict(
    # Control voltage
    VCTRL = dict(
        AO_N_CHANNEL = 0,
        AO_P_CHANNEL = 1,
        ),
    # Voltage monitor
    VMON = dict(
        AI_CHANNEL = 0,
        GAIN = -100.62, 
        OFFSET = 0,
        ),
    # Current monitor
    IMON = dict(
        AI_CHANNEL = 1,
        GAIN = -1501.8,  # Ryan's board + 10.0 ohm current sense resistor
        # GAIN = -4520.4,  # Ryan's board + 30.1 ohm current sense resistor
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
    filename = user_config_dir("rxlab-sis-bias.config")
else:
    filename = args.configfile

# Save config file
with open(filename, 'w') as fout:
    json.dump(config, fout, indent=4)
print(f"\nConfiguration file saved to: {filename}\n")