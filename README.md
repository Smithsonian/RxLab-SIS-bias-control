Receiver Lab: SIS Bias Control
==============================

*Control the SIS bias via the MCC DAQ device*

This package contains code to control the bias voltage of an SIS mixer using an MCC DAQ device. It can also be used to read the voltage and current monitors in order to record a current-voltage (I-V) curve.

This package is used by the Receiver Lab (Submillimeter Array) to test SIS mixers in the lab.

Installation
------------

Install the [MCC Universal Library for Linux (uldaq)](https://github.com/mccdaq/uldaq). For macOS:
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

Then you can download this package:
```bash
git clone https://github.com/Smithsonian/RxLab-SIS-bias
cd RxLab-SIS-bias
python3 -m pip install -e .
```

Getting Started
---------------

The code to control the SIS bias via the MCC DAQ device is contained within `sisbias/`. 

There is also an interactive command-line interface: `sisbias`.
