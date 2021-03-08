SIS Bias Control
================

Control SIS bias via MCC DAQ device.

Installation for macOS
----------------------

```bash
# Install MCC Universal Library for Linux (uldaq)
xcode-select --install
ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
brew install libusb
curl -L -O https://github.com/mccdaq/uldaq/releases/download/v1.2.0/libuldaq-1.2.0.tar.bz2
tar -xvjf libuldaq-1.2.0.tar.bz2
cd libuldaq-1.2.0
./configure && make
sudo make install
pip install uldaq

# Download SIS bias control script
git clone https://github.com/garrettj403/SIS-bias-control
```