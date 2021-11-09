Receiver Lab: SIS Bias Control
==============================

*Control the SIS bias board using an MCC DAQ device* 

This package is used by the Receiver Lab (Submillimeter Array) to test SIS mixers in the lab.

Please see [the project wiki](https://github.com/Smithsonian/RxLab-SIS-bias/wiki) for more information.

Getting Started
---------------

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

You can then download and install this package:
```bash
git clone https://github.com/Smithsonian/RxLab-SIS-bias
cd RxLab-SIS-bias
python3 -m pip install -e .
```

Useage
------

All of the code for interfacing with the MCC DAQ device + SIS bias board is contained in [the `sisbias/` directory](https://github.com/Smithsonian/RxLab-SIS-bias/tree/main/sisbias), with the most important code in [the `control.py` file](https://github.com/Smithsonian/RxLab-SIS-bias/blob/main/sisbias/control.py).

I have also included [an interactive interface](https://github.com/Smithsonian/RxLab-SIS-bias/blob/main/bin/sisbias) to make it easier to perform measurements in the lab. You can start this environment by typing `sisbias` in the terminal.

Please see [the project wiki](https://github.com/Smithsonian/RxLab-SIS-bias/wiki) for more information.
