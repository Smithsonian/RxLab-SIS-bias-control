"""Control SIS bias via MCC DAQ device."""

import json
import time

import matplotlib.pyplot as plt
import numpy as np
import uldaq
from appdirs import user_config_dir
from uldaq import (AiInputMode, AInFlag, AInScanFlag, AOutFlag, AOutScanFlag,
                   DaqDevice, InterfaceType, Range, ScanOption, ScanStatus,
                   create_float_buffer, get_daq_device_inventory)

# DAQ
INTERFACE_TYPE = InterfaceType.USB

# DAQ output channels
AO_RANGE = Range.UNI5VOLTS
AO_FLAG = AOutFlag.DEFAULT
SCAN_OPTIONS = ScanOption.CONTINUOUS
SCAN_FLAGS = AOutScanFlag.DEFAULT

# DAQ input Channels
AI_MODE = AiInputMode.DIFFERENTIAL
AI_RANGE = Range.BIP5VOLTS
AI_FLAG = AInFlag.DEFAULT
AI_SCAN_FLAG = AInScanFlag.DEFAULT


class SISBias:
    """Class for SIS bias control."""
    
    def __init__(self, config_file=None):
        
        # Read parameters file
        if config_file is None:
            config_file = user_config_dir("rxlab-sis-bias")
        with open(config_file) as _fin:
            self.params = json.load(_fin)

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
            msg = '\nPlease select a DAQ device (between 0 and ' + str(number_of_devices - 1) + '): '
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
            raise RuntimeError('Error: The DAQ device does not support analog input')
                               
        # Verify the specified DAQ device supports analog output
        if self.ao_device is None:
            raise RuntimeError('Error: The DAQ device does not support analog output')
                
        # Verify the device supports hardware pacing for analog output
        self.ao_info = self.ao_device.get_info()
        if not self.ao_info.has_pacer():
            raise RuntimeError('Error: The DAQ device does not support paced analog output')
                               
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

    def __repr__(self):

        return f"SIS bias control via MCC DAQ device: {self.daq_name}"

    def __str__(self):

        return self.__repr__()

    # Set control voltage ------------------------------------------------ ###

    def set_control_voltage(self, voltage, vmax=3, verbose=False):
        """Set control voltage to a constant value (no sweep).

        Uses two channels to create a differential output (to allow negative
        control voltages).

        Args:
            voltage: control voltage

        """

        # Stop current scan
        self.update_ao_scan_status()
        if self._ao_scan_status == ScanStatus.RUNNING:
            self.ao_device.scan_stop()

        # Check if within limits
        msg = "\t\t** warning: control voltage beyond limits ** "
        if voltage > vmax:
            print(msg)
            voltage = vmax
        elif voltage < -vmax:
            print(msg)
            voltage = -vmax

        if voltage >= 0:
            self.ao_device.a_out(self.params['VCTRL']['AO_N_CHANNEL'], AO_RANGE, AO_FLAG, 0.0)
            self.ao_device.a_out(self.params['VCTRL']['AO_P_CHANNEL'], AO_RANGE, AO_FLAG, voltage)
        else:
            self.ao_device.a_out(self.params['VCTRL']['AO_N_CHANNEL'], AO_RANGE, AO_FLAG, -voltage)
            self.ao_device.a_out(self.params['VCTRL']['AO_P_CHANNEL'], AO_RANGE, AO_FLAG, 0.0)

        if verbose:
            print(f"Control voltage set to {voltage:.1f} V")
    
    def set_bias_voltage(self, vbias_target, dvctrl=0.1, vctrl_start=0, sleep_time=0.1, iterations=3, verbose=False):

        # Starting value
        vctrl = vctrl_start
        self.set_control_voltage(vctrl, vmax=2)
        time.sleep(sleep_time)

        # Iterate to find control voltage
        for _ in range(iterations):

            # Read bias voltage
            _vbias1 = np.zeros(100)
            for i in range(100):
                _vbias1[i] = self.read_voltage()
            vbias1 = np.mean(_vbias1)
            if verbose:
                print("\tBias voltage:  {:6.2f} mV".format(vbias1))
            
            # Calculate derivative
            self.set_control_voltage(vctrl + dvctrl, vmax=2)
            time.sleep(sleep_time)
            _vbias2 = np.zeros(100)
            for i in range(100):
                _vbias2[i] = self.read_voltage()
            vbias2 = np.mean(_vbias2)
            der = (vbias2 - vbias1) / dvctrl

            # Update control voltage
            error = vbias1 - vbias_target
            vctrl -= error / der 
            self.set_control_voltage(vctrl, vmax=2)
            time.sleep(sleep_time)

        # Read final value
        vbias = self.read_voltage()
        if verbose:
            print("\tBias voltage:  {:6.2f} mV\n".format(vbias))

        return vbias, vctrl

    def sweep_control_voltage(self, vmin=-1, vmax=1, npts=1000, sweep_period=5.0, verbose=True):
        """Sweep control voltage (triangle wave).

        Uses two channels to create a differential output (to allow negative
        control voltages).

        Args:
            vmin (float): minimum voltage
            vmax (float): maximum voltage

        """

        # TODO: make vmin, vmax, period, sample_rate into properties??

        sample_period = sweep_period / npts
        sample_frequency = 1 / sample_period
        samples_per_period = int(2 * npts)
        samples_per_channel = int(npts)

        # # Stop current scan
        # self.update_ao_scan_status()
        # if self._ao_scan_status == ScanStatus.RUNNING:
        #     self.ao_device.scan_stop()

        # Build control voltage (triangle wave)
        vctrl_up = np.linspace(vmin, vmax, samples_per_channel // 2)
        vctrl_down = vctrl_up[::-1]
        vctrl = np.r_[vctrl_up, vctrl_down]
        vctrl = np.roll(vctrl, int(len(vctrl) / 4))
        vctrl_p = np.zeros_like(vctrl)
        vctrl_n = np.zeros_like(vctrl)
        vctrl_p[vctrl > 0] = vctrl[vctrl > 0]
        vctrl_n[vctrl < 0] = -vctrl[vctrl < 0]
        vctrl_weave = np.empty(len(vctrl_n) + len(vctrl_p), dtype=float)
        vctrl_weave[::2] = vctrl_n
        vctrl_weave[1::2] = vctrl_p

        # Start analog output scan
        self._sweep_control_voltage(vctrl_weave, sweep_period=sweep_period, sample_frequency=sample_frequency, verbose=verbose)

    def pulse_control_voltage(self, vmin=-1, vmax=1, npts=1000, sweep_period=5.0, verbose=True):
        """Pulse control voltage (square wave).

        Uses two channels to create a differential output (to allow negative
        control voltages).

        Args:
            vmin (float): minimum voltage
            vmax (float): maximum voltage

        """

        sample_period = sweep_period / npts
        sample_frequency = 1 / sample_period
        samples_per_period = int(2 * npts)
        samples_per_channel = int(npts)

        # Stop current scan
        # self.update_ao_scan_status()
        # if self._ao_scan_status == ScanStatus.RUNNING:
        #     self.ao_device.scan_stop()

        # Build control voltage (square wave)
        vctrl_p = vmax * np.r_[np.ones(samples_per_channel//2),
                               np.zeros(samples_per_channel//2)]
        vctrl_n = vmin * np.r_[np.zeros(samples_per_channel//2),
                               np.ones(samples_per_channel//2)]
        vctrl = np.empty(len(vctrl_n) + len(vctrl_p), dtype=float)
        vctrl[::2] = vctrl_n
        vctrl[1::2] = vctrl_p

        # Start analog output scan
        self._sweep_control_voltage(vctrl, sweep_period=sweep_period, sample_frequency=sample_frequency, verbose=verbose)

    def _sweep_control_voltage(self, voltage, sweep_period=5.0, sample_frequency=1000, verbose=True):
        """Sweep control voltage.

        Args:
            voltage (np.ndarray): voltage array for buffer
            sweep_period (float): period
            sample_frequency (int): sample frequency

        """

        num_channels = 2
        samples_per_period = int(sample_frequency * sweep_period)
        samples_per_channel = int(samples_per_period)

        # Stop current scan
        self.update_ao_scan_status()
        if self._ao_scan_status == ScanStatus.RUNNING:
            self.ao_device.scan_stop()

        # Create output buffer for control voltage
        output_buffer = create_float_buffer(2, samples_per_channel)

        # Min/max value
        control_signal = voltage[1::2] - voltage[::2]
        vmin = control_signal.min()
        vmax = control_signal.max()
        npts = len(control_signal)

        # Fill buffer with data
        for i in range(len(voltage)):
            output_buffer[i] = voltage[i]

        # Start the output scan.
        vctrl_n_chan = self.params['VCTRL']['AO_N_CHANNEL']
        vctrl_p_chan = self.params['VCTRL']['AO_P_CHANNEL']
        rate = self.ao_device.a_out_scan(vctrl_n_chan, vctrl_p_chan,
                                         AO_RANGE, samples_per_channel, 
                                         sample_frequency, SCAN_OPTIONS, 
                                         SCAN_FLAGS, output_buffer)

        if verbose:
            print("\n\tSweep control voltage:")
            print(f'\t\t{self.daq_name}: ready')
            print(f'\t\tDAC range:           {AO_RANGE.name}')
            print(f'\t\tVoltage range:       {vmin:.1f} to {vmax:.1f} V')
            print(f'\t\tVoltage points:      {npts}')
            print(f'\t\tSweep frequency:     {1 / sweep_period:.1f} Hz')
            print(f'\t\tSweep period:        {sweep_period:.1f} s')
            print(f'\t\tSampling frequency:  {sample_frequency:.1f} Hz')
            print(f'\t\tSampling frequency:  {rate:.1f} Hz (actual)')

    # Read voltage & current monitor ------------------------------------- ###

    def read_voltage(self):
        """Read voltage monitor.

        Returns:
            voltage monitor in [mV]

        """
        
        # Stop current scan
        self.update_ai_scan_status()
        if self._ai_scan_status == ScanStatus.RUNNING:
            self.ai_device.scan_stop()

        return self._read_analog(self.params['VMON']['AI_CHANNEL']) / self.params['VMON']['GAIN'] * 1e3  - self.params['VMON']['OFFSET']
        
    def read_current(self):
        """Read current monitor.

        Returns:
            current monitor in [uA]

        """
        
        # Stop current scan
        self.update_ai_scan_status()
        if self._ai_scan_status == ScanStatus.RUNNING:
            self.ai_device.scan_stop()

        return self._read_analog(self.params['IMON']['AI_CHANNEL']) / self.params['IMON']['GAIN'] * 1e6 - self.params['IMON']['OFFSET']
    
    def read_ifpower(self):
        """Read IF power from power meter/detector.

        Returns:
            IF power in A.U.

        """

        # Stop current scan
        self.update_ai_scan_status()
        if self._ai_scan_status == ScanStatus.RUNNING:
            self.ai_device.scan_stop()

        return self._read_analog(self.params['PIF']['AI_CHANNEL']) - self.params['PIF']['OFFSET']

    def read_iv_curve(self, debug=False):
        """Read I-V curve.

        Returns:
            voltage in V, current in A, IF power in A.U.

        """

        # Analog input from voltage / current monitors
        data_in = list(self.analog_input)
        voltage, current, ifpower = np.array(data_in[::3]), np.array(data_in[1::3]), np.array(data_in[2::3])

        if debug:
            print("Raw voltage range: {:.2f} to {:.2f} V".format(voltage.min(), voltage.max()))
            print("Raw current range: {:.2f} to {:.2f} V".format(current.min(), current.max()))

        # Calibrate and correct offset
        voltage = voltage / self.params['VMON']['GAIN'] - self.params['VMON']['OFFSET']
        current = current / self.params['IMON']['GAIN'] - self.params['IMON']['OFFSET']
        ifpower = ifpower - self.params['PIF']['OFFSET']

        return voltage, current, ifpower

    def _read_analog(self, channel):
        """Read analog input channel.

        Args:
            channel: channel number

        Returns:
            analog input

        """

        return self.ai_device.a_in(channel, AI_MODE, AI_RANGE, AI_FLAG)

    def start_iv_monitor_scan(self, npts=1000, sweep_period=5.0, verbose=True):
        """Scan voltage & current monitors.

        Args:

        """

        sample_period = sweep_period / npts
        sample_frequency = 1 / sample_period
        samples_per_period = int(npts * 3)
        samples_per_channel = int(npts)

        # Stop current scan
        self.update_ai_scan_status()
        if self._ai_scan_status == ScanStatus.RUNNING:
            self.ai_device.scan_stop()

        # Allocate a buffer to receive the data.
        self.analog_input = create_float_buffer(3, samples_per_channel)

        # Start the acquisition.
        rate = self.ai_device.a_in_scan(self.params['VMON']['AI_CHANNEL'], 
                                        self.params['PIF']['AI_CHANNEL'],
                                        AI_MODE, AI_RANGE, samples_per_channel,
                                        sample_frequency, SCAN_OPTIONS,
                                        AI_SCAN_FLAG, self.analog_input)

        if verbose:
            print("\n\tReading voltage & current monitors:")
            print(f'\t\t{self.daq_name}: ready')
            print(f'\t\tADC range:           {AI_RANGE.name}')
            print(f'\t\tNumber of samples:   {samples_per_channel:d}')
            print(f'\t\tSweep frequency:     {1/sweep_period:.1f} Hz')
            print(f'\t\tSweep period:        {sweep_period:.1f} s')
            print(f'\t\tSampling frequency:  {sample_frequency:.1f} Hz')
            print(f'\t\tSampling frequency:  {rate:.1f} Hz (actual)')

    # Plot --------------------------------------------------------------- ###

    def plot(self):
        """Plot I-V curve."""

        voltage, current, power = self.read_iv_curve()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
        ax1.plot(voltage*1e3, current*1e6, 'ko', alpha=0.2, ms=1)
        ax1.set_xlabel("Voltage (mV)")
        ax1.set_ylabel("Current (uA)")

        ax2.plot(voltage*1e3, power, 'ko', alpha=0.2, ms=1)
        ax2.set_xlabel("Voltage (mV)")
        ax2.set_ylabel("Power (au)")

        plt.show()

    # Scan status -------------------------------------------------------- ###
    
    def ao_scan_status(self):
        """Update and return scan status."""

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
        """Update and return scan status."""

        self._ai_scan_status, self._ai_transfer_status = self.ai_device.get_scan_status()

    # Stop --------------------------------------------------------------- ###
    
    def close(self):
        """Stop DAQ device and close all connections."""
        
        try:
            print("\nClosing connection to DAQ device ... ", end='')
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
            print("\nDevice already disconnected\n")

    def stop(self):
        """Stop DAQ device and close all connections."""
        
        self.close()
