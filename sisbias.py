"""Control SIS bias via MCC DAQ device."""

import time
import numpy as np
import matplotlib.pyplot as plt

import uldaq
from uldaq import (get_daq_device_inventory, DaqDevice, InterfaceType, 
                   AOutFlag, AiInputMode, AInFlag, Range, ScanStatus,
                   create_float_buffer, AOutScanFlag, ScanOption, AInScanFlag)


# DAQ
INTERFACE_TYPE = InterfaceType.USB

# DAQ output channels
AO_RANGE = Range.UNI5VOLTS
AO_FLAG = AOutFlag.DEFAULT
VCTRL_N_CHANNEL = 0
VCTRL_P_CHANNEL = 1
SCAN_OPTIONS = ScanOption.CONTINUOUS
SCAN_FLAGS = AOutScanFlag.DEFAULT

# DAQ input Channels
AI_MODE = AiInputMode.DIFFERENTIAL
AI_RANGE = Range.BIP5VOLTS
AI_FLAG = AInFlag.DEFAULT
AI_SCAN_FLAG = AInScanFlag.DEFAULT
VMON_AI_CHANNEL = 0
IMON_AI_CHANNEL = 1
VMON_CONV_MV = 5
IMON_CONV_UA = 5 / 10 * 1000


class SISBias:
    """Class for SIS bias control."""
    
    def __init__(self):
        
        # Get all available DAQ devices
        devices = get_daq_device_inventory(INTERFACE_TYPE)
        number_of_devices = len(devices)

        # Verify at least one DAQ device is detected
        if number_of_devices == 0:
            raise RuntimeError('Error: No DAQ devices found')
            
        # Print all available devices
        print('\nFound', number_of_devices, 'DAQ device(s):')
        for i in range(number_of_devices):
            print('  [', i, '] ', devices[i].product_name, ' (',
                  devices[i].unique_id, ')', sep='')
                
        # Choose DAQ device
        if number_of_devices == 1:
            descriptor_index = 0
        else:
            msg = '\nPlease select a DAQ device, enter a number' + \
                  ' between0 and ' + str(number_of_devices - 1) + ': '
            descriptor_index = input(msg)
            descriptor_index = int(descriptor_index)
            if descriptor_index not in range(number_of_devices):
                raise RuntimeError('Error: Invalid descriptor index')
            
        # Create the DAQ device object
        self.daq_device = DaqDevice(devices[descriptor_index])
        self.ao_device = self.daq_device.get_ao_device()
        self.ai_device = self.daq_device.get_ai_device()

        # Verify the specified DAQ device supports analog input
        if self.ai_device is None:
            raise RuntimeError('Error: The DAQ device does not support '
                               'analog input')
                               
        # Verify the specified DAQ device supports analog output
        if self.ao_device is None:
            raise RuntimeError('Error: The DAQ device does not support' 
                               'analog output')
                
        # Verify the device supports hardware pacing for analog output
        self.ao_info = self.ao_device.get_info()
        if not self.ao_info.has_pacer():
            raise RuntimeError('Error: The DAQ device does not support' 
                               ' paced analog output')
                               
        # Establish a connection to the device.
        self.desc = self.daq_device.get_descriptor()
        self.daq_name = "{} ({})".format(self.desc.dev_string, self.desc.unique_id)
        print(f'\nConnecting to {self.daq_name} ... ', end=" ")
        self.daq_device.connect(connection_code=0)
        print("done\n")
        
        # Initialize scan status
        self._ao_scan_status, self._ao_transfer_status = self.ao_device.get_scan_status()
        self._ai_scan_status, self._ai_transfer_status = self.ai_device.get_scan_status()
        self.analog_input = None

    # Properties --------------------------------------------------------- ###

    # Set control voltage ------------------------------------------------ ###

    def set_control_voltage(self, voltage):
        """Set control voltage to a constant value (not swept).

        Uses two channels to create a differential output (to allow negative
        control voltages).

        Args:
            voltage: control voltage

        """
        
        if voltage >= 0:
            self.ao_device.a_out(VCTRL_N_CHANNEL, AO_RANGE, AO_FLAG, 0.0)
            self.ao_device.a_out(VCTRL_P_CHANNEL, AO_RANGE, AO_FLAG, voltage)
        else:
            self.ao_device.a_out(VCTRL_P_CHANNEL, AO_RANGE, AO_FLAG, 0.0)
            self.ao_device.a_out(VCTRL_N_CHANNEL, AO_RANGE, AO_FLAG, -voltage)
    
    def sweep_control_voltage(self, vmin=-1.0, vmax=1.0, period=5.0, sample_rate=1000):
        """Sweep control voltage (triangle wave).

        Uses two channels to create a differential output (to allow negative
        control voltages).

        Args:
            vmin (float): minimum voltage
            vmax (float): maximum voltage
            period (float): period of sweep
            sample_rate (int): sample rate

        """

        # TODO: make vmin, vmax, period, sample_rate into properties

        # Stop current scan
        self.update_ao_scan_status()
        if self._ao_scan_status == ScanStatus.RUNNING:
            self.ao_device.scan_stop()

        num_channels = 2
        samples_per_period = int(sample_rate * period)
        samples_per_channel = int(samples_per_period / num_channels)

        # Build control voltage (triangle wave)
        vctrl = np.linspace(vmin, vmax, samples_per_channel // 2)
        vctrl = np.r_[vctrl, vctrl[::-1]]
        vctrl_p = np.zeros_like(vctrl)
        vctrl_n = np.zeros_like(vctrl)
        vctrl_p[vctrl > 0] = vctrl[vctrl > 0]
        vctrl_n[vctrl < 0] = -vctrl[vctrl < 0]
        vctrl = np.empty(len(vctrl_n) + len(vctrl_p), dtype=float)
        vctrl[::2] = vctrl_n
        vctrl[1::2] = vctrl_p

        # Start analog output scan
        self._scan_control_voltage(vctrl, period=period, sample_rate=sample_rate)

    def pulse_control_voltage(self, vmin=0.0, vmax=1.0, period=0.1, sample_rate=10000):
        """Pulse control voltage (square wave).

        Uses two channels to create a differential output (to allow negative
        control voltages).

        Args:
            vmin (float): minimum voltage
            vmax (float): maximum voltage
            period (float): period of sweep
            sample_rate (int): sample rate

        """

        # Stop current scan
        self.update_ao_scan_status()
        if self._ao_scan_status == ScanStatus.RUNNING:
            self.ao_device.scan_stop()

        num_channels = 2
        samples_per_period = int(sample_rate * period)
        samples_per_channel = int(samples_per_period / num_channels)

        # Build control voltage (square wave)
        vctrl_p = vmax * np.r_[np.ones(samples_per_channel//2),
                               np.zeros(samples_per_channel//2)]
        vctrl_n = vmin * np.r_[np.zeros(samples_per_channel//2),
                               np.ones(samples_per_channel//2)]
        vctrl = np.empty(len(vctrl_n) + len(vctrl_p), dtype=float)
        vctrl[::2] = vctrl_n
        vctrl[1::2] = vctrl_p

        # Start analog output scan
        self._scan_control_voltage(vctrl, period=period, sample_rate=sample_rate)

    def _scan_control_voltage(self, voltage, period=5.0, sample_rate=1000):
        """Setup control voltage scan.

        Args:
            voltage (np.ndarray): voltage array for buffer
            period (float): period
            sample_rate (int): sample rate

        """

        # Stop current scan
        self.update_ao_scan_status()
        if self._ao_scan_status == ScanStatus.RUNNING:
            self.ao_device.scan_stop()

        num_channels = 2
        samples_per_period = int(sample_rate * period)
        samples_per_channel = int(samples_per_period / num_channels)

        # Create output buffer for control voltage
        output_buffer = create_float_buffer(num_channels, samples_per_channel)

        # Fill buffer with data
        for i in range(len(voltage)):
            output_buffer[i] = voltage[i]

        # Start the output scan.
        rate = self.ao_device.a_out_scan(VCTRL_N_CHANNEL, VCTRL_P_CHANNEL,
                                         AO_RANGE, samples_per_channel, sample_rate,
                                         SCAN_OPTIONS, SCAN_FLAGS, output_buffer)

        print("\nSweeping control voltage:")
        print(f'    {self.daq_name}: ready')
        print(f'    Sweep frequency: {1 / period} Hz')
        print(f'    Range: {AO_RANGE.name}')
        print(f'    Samples per channel: {samples_per_channel}')
        print(f'    Sample Rate: {sample_rate} Hz')
        print(f'    Actual sample rate: {rate} Hz\n')

    # Read voltage & current monitor ------------------------------------- ###

    def read_voltage(self):
        """Read voltage monitor.

        Returns:
            voltage monitor in [mV]

        """
        
        return self._read_analog(VMON_AI_CHANNEL) * VMON_CONV_MV
        
    def read_current(self):
        """Read current monitor.

        Returns:
            current monitor in [uA]

        """
        
        return self._read_analog(IMON_AI_CHANNEL) * IMON_CONV_UA
    
    def _read_analog(self, channel):
        """Read analog input channel.

        Args:
            channel: channel number

        Returns:
            analog input

        """

        return self.ai_device.a_in(channel, AI_MODE, AI_RANGE, AI_FLAG)

    def start_iv_monitor_scan(self, period=5.0, sample_rate=1000):
        """Scan voltage & current monitors.

        Args:
            period (float): period
            sample_rate (int): sample rate

        """

        # Stop current scan
        self.update_ai_scan_status()
        if self._ai_scan_status == ScanStatus.RUNNING:
            self.ai_device.scan_stop()

        num_channels = 2
        samples_per_period = int(sample_rate * period)
        samples_per_channel = int(samples_per_period / num_channels)

        # Allocate a buffer to receive the data.
        self.analog_input = create_float_buffer(num_channels, samples_per_channel)

        # Start the acquisition.
        rate = self.ai_device.a_in_scan(VMON_AI_CHANNEL, IMON_AI_CHANNEL,
                                        AI_MODE, AI_RANGE, samples_per_channel,
                                        sample_rate, SCAN_OPTIONS,
                                        AI_SCAN_FLAG, self.analog_input)

        print("\nReading voltage & current monitors:")
        print(f'    {self.daq_name}: ready')
        print(f'    Sweep frequency: {1 / period} Hz')
        print(f'    Range: {AI_RANGE.name}')
        print(f'    Samples per channel: {samples_per_channel}')
        print(f'    Sample Rate: {sample_rate} Hz')
        print(f'    Actual sample rate: {rate} Hz\n')

    # Plot --------------------------------------------------------------- ###

    def plot(self):

        # TODO: fix

        data_in = list(self.analog_input)
        v, i = data_in[::2], data_in[1::2]

        plt.figure()
        plt.ion()
        plt.plot(v, i)
        plt.show()

    # Misc --------------------------------------------------------------- ###
    
    def ao_scan_status(self):
        """Return scan status."""

        self.update_ao_scan_status()
        return self._ao_scan_status

    def update_ao_scan_status(self):
        """Update scan status."""

        self._ao_scan_status, self._ao_transfer_status = self.ao_device.get_scan_status()

    def ai_scan_status(self):
        """Return scan status."""

        self.update_ai_scan_status()
        return self._ai_scan_status

    def update_ai_scan_status(self):
        """Update scan status."""

        self._ai_scan_status, self._ai_transfer_status = self.ai_device.get_scan_status()

    def stop(self):
        """Stop DAQ device and close all connections."""
        
        self.close()
    
    def close(self):
        """Stop DAQ device and close all connections."""
        
        try:
            print("Closing connection to DAQ device ... ", end='')
            if self.daq_device:
                self.update_ao_scan_status()
                # Stop the scan
                if self._ao_scan_status == ScanStatus.RUNNING:
                    self.ao_device.scan_stop()
                self.set_control_voltage(0)
                # Disconnect from the DAQ device
                if self.daq_device.is_connected():
                    self.daq_device.disconnect()
                # Release the DAQ device resource
                self.daq_device.release()
            print("done\n")
        except uldaq.ul_exception.ULException:
            print("device already disconnected\n")


if __name__ == "__main__":
        
    bias = SISBias()
    # bias.sweep_voltage(-2, 2, 5)
    # time.sleep(0.1)

    # fig, ax = plt.subplots()
    # ax.set_xlabel("Voltage (mV)")
    # ax.set_ylabel("Current (uA)")
    # ax.set_title("SIS bias control")

    # npts = 5000
    # voltage, current = np.empty(npts), np.empty(npts)
    # for i in range(npts):
    #     voltage[i] = bias.read_voltage()
    #     current[i] = bias.read_current()
    # plt.plot(voltage, current, 'ko', alpha=0.2, ms=1)
    # plt.show()

    # bias.stop()
