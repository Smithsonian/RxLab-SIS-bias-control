SIS Bias Control
================

Control SIS bias via MCC DAQ device.

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