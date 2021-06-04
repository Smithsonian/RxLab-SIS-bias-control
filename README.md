Receiver Lab: SIS Bias Control
==============================

*Control SIS bias via MCC DAQ device*

This package contains code to control the bias voltage of an SIS mixer using an MCC DAQ device. It can also be used to read the voltage and current monitors in order to record a current-voltage (I-V) curve.

This package is used by the Receiver Lab (Submillimeter Array) to test SIS mixers in the lab.

Installation
------------

Install [MCC Universal Library for Linux (uldaq)](https://github.com/mccdaq/uldaq). For macOS:
```bash
# Install X-code tools
xcode-select --install

# Install homebrew (package manager)
ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

# Install libusb
brew install libusb

# Install uldaq library
curl -L -O https://github.com/mccdaq/uldaq/releases/download/v1.2.0/libuldaq-1.2.0.tar.bz2
tar -xvjf libuldaq-1.2.0.tar.bz2
cd libuldaq-1.2.0
./configure && make
sudo make install

# Install uldaq python package
pip install uldaq
```
See [uldaq webpage](https://github.com/mccdaq/uldaq) for other platforms.

Download this package:
```bash
git clone https://github.com/Smithsonian/RxLab-SIS-bias
cd RxLab-SIS-bias
```

Getting Started
---------------

The code to control the SIS bias via the MCC DAQ device is contained within `sisbias.py`. 

I also provide an interactive script `main.py` to make things easier.

```python
python main.py
```

With `main.py` running, type in "HELP" or "H" to see the available commands.

```bash
>> HELP
Available commands:
	HELP or H: Print help
	SWEEP or START or S: Sweep control voltage (triangle wave)
	PULSE: Pulse control voltage (square wave)
	VSET: Set constant control voltage
	VMON: Read voltage monitor
	IMON: Read current monitor
	INFO: Print all parameters
	STATUS: Print scan status
	PLOT or P: Plot I-V curve
	CLEAR or C: Clear all plots
	STOP or EXIT or Q: Close connection

Available parameters:
	VMIN <value>: Minimum control voltage for sweep or pulse, in [V]
	VMAX <value>: Maximum control voltage for sweep or pulse, in [V]
	PERIOD <value>: Period of sweep or pulse, in [s]
	SAMPLE_RATE <value>: Sample rate for control voltage sweep or pulse, in [Hz]
```