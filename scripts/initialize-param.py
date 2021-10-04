"""Initialize parameters file for SIS bias control."""

import json

from appdirs import user_config_dir
import argparse

# Grab arguments
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--paramfile", type=str, help="Param file", default=None)
args = parser.parse_args()


# Default parameters
params = dict(
    VMIN=-1,
    VMAX=1,
    PERIOD=0.2,
    NPTS=2000,
    FREQ=0,
    NJUNC=3,
    IOFFSET=0,
    IFOFFSET=0,
    LNA='on',
    IFCORR=1,
)


# Location of param file
if args.paramfile is None:
    filename = user_config_dir("rxlab-sis-bias.param")
else:
    filename = args.configfile

# Save config file
with open(filename, 'w') as fout:
    json.dump(params, fout, indent=4)

print(f"Parameter file saved to: {filename}")
