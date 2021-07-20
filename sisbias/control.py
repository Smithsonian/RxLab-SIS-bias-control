"""Control SIS bias via MCC DAQ device."""

import json
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
SCAN_OPTIONS = ScanOption.CONTINUOUS
SCAN_FLAGS = AOutScanFlag.DEFAULT

# DAQ input Channels
AI_MODE = AiInputMode.DIFFERENTIAL
AI_RANGE = Range.BIP2VOLTS
AI_FLAG = AInFlag.DEFAULT
AI_SCAN_FLAG = AInScanFlag.DEFAULT


class SISBias:
    """Class for SIS bias control."""
    
    def __init__(self, param_file='params.json'):
        
        # Read parameters file
        with open(param_file) as _fin:
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

    def set_control_voltage(self, voltage, vmax=3):
        """Set control voltage to a constant value (not swept).

        Uses two channels to create a differential output (to allow negative
        control voltages).

        Args:
            voltage: control voltage

        """

        # Stop current scan
        self.update_ao_scan_status()
        if self._ao_scan_status == ScanStatus.RUNNING:
            self.ao_device.scan_stop()

        # Check if in limits
        if voltage > vmax:
            voltage = vmax
        elif voltage < -vmax:
            voltage = -vmax

        if voltage >= 0:
            self.ao_device.a_out(self.params['VCTRL_N_CHANNEL'], AO_RANGE, AO_FLAG, 0.0)
            self.ao_device.a_out(self.params['VCTRL_P_CHANNEL'], AO_RANGE, AO_FLAG, voltage)
        else:
            self.ao_device.a_out(self.params['VCTRL_P_CHANNEL'], AO_RANGE, AO_FLAG, 0.0)
            self.ao_device.a_out(self.params['VCTRL_N_CHANNEL'], AO_RANGE, AO_FLAG, -voltage)
    
    def sweep_control_voltage(self, vmin=-1.0, vmax=1.0, npts=1000, sweep_period=5.0, verbose=True):
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

        # Stop current scan
        self.update_ao_scan_status()
        if self._ao_scan_status == ScanStatus.RUNNING:
            self.ao_device.scan_stop()

        # Build control voltage (triangle wave)
        vctrl_up = np.linspace(-vmax, -vmin, samples_per_channel // 2)
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

    # def pulse_control_voltage(self, vmin=0.0, vmax=1.0, period=0.1, sample_rate=10000):
    #     """Pulse control voltage (square wave).

    #     Uses two channels to create a differential output (to allow negative
    #     control voltages).

    #     Args:
    #         vmin (float): minimum voltage
    #         vmax (float): maximum voltage
    #         period (float): period of sweep
    #         sample_rate (int): sample rate

    #     """

    #     # Stop current scan
    #     self.update_ao_scan_status()
    #     if self._ao_scan_status == ScanStatus.RUNNING:
    #         self.ao_device.scan_stop()

    #     num_channels = 2
    #     samples_per_period = int(sample_rate * period)
    #     samples_per_channel = int(samples_per_period / num_channels)

    #     # Build control voltage (square wave)
    #     vctrl_p = vmax * np.r_[np.ones(samples_per_channel//2),
    #                            np.zeros(samples_per_channel//2)]
    #     vctrl_n = vmin * np.r_[np.zeros(samples_per_channel//2),
    #                            np.ones(samples_per_channel//2)]
    #     vctrl = np.empty(len(vctrl_n) + len(vctrl_p), dtype=float)
    #     vctrl[::2] = vctrl_n
    #     vctrl[1::2] = vctrl_p

    #     # Start analog output scan
    #     self._sweep_control_voltage(vctrl, period=period, sample_rate=sample_rate)

    def _sweep_control_voltage(self, voltage, sweep_period=5.0, sample_frequency=1000, verbose=True):
        """Sweep control voltage.

        Args:
            voltage (np.ndarray): voltage array for buffer
            sweep_period (float): period
            sample_frequency (int): sample frequency

        """

        # Stop current scan
        self.update_ao_scan_status()
        if self._ao_scan_status == ScanStatus.RUNNING:
            self.ao_device.scan_stop()

        num_channels = 2
        samples_per_period = int(sample_frequency * sweep_period)
        samples_per_channel = int(samples_per_period)

        # Create output buffer for control voltage
        output_buffer = create_float_buffer(2, samples_per_channel)

        # Fill buffer with data
        for i in range(len(voltage)):
            output_buffer[i] = voltage[i]

        # Start the output scan.
        rate = self.ao_device.a_out_scan(self.params['VCTRL_N_CHANNEL'], 
                                         self.params['VCTRL_P_CHANNEL'],
                                         AO_RANGE, samples_per_channel, sample_frequency,
                                         SCAN_OPTIONS, SCAN_FLAGS, output_buffer)

        if verbose:
            print("\n\tSweeping control voltage:")
            print(f'\t\t{self.daq_name}: ready')
            print(f'\t\tRange: {AO_RANGE.name}')
            print(f'\t\tSweep period: {sweep_period:.1f} s')
            print(f'\t\tSweep frequency: {1 / sweep_period:.1f} Hz')
            print(f'\t\tSamples per channel: {samples_per_channel:d}')
            print(f'\t\tSample Rate: {sample_frequency:.1f} Hz')
            print(f'\t\tActual sample rate: {rate:.1f} Hz\n')

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

        return (self._read_analog(self.params['VMON_AI_CHANNEL']) - self.params['VMON']['OFFSET']) / self.params['VMON']['GAIN'] * 1e3
        
    def read_current(self):
        """Read current monitor.

        Returns:
            current monitor in [uA]

        """
        
        # Stop current scan
        self.update_ai_scan_status()
        if self._ai_scan_status == ScanStatus.RUNNING:
            self.ai_device.scan_stop()

        return (self._read_analog(self.params['IMON_AI_CHANNEL']) - self.params['IMON']['OFFSET']) / self.params['IMON']['GAIN'] * 1e6
    
    def read_ifpower(self):
        """Read IF power from power meter/detector.

        Returns:
            IF power in A.U.

        """

        # Stop current scan
        self.update_ai_scan_status()
        if self._ai_scan_status == ScanStatus.RUNNING:
            self.ai_device.scan_stop()

        return self._read_analog(self.params['PIF_AI_CHANNEL'])

    def read_iv_curve(self, debug=False):
        """Read I-V curve.

        Returns:
            voltage in V and current in A

        """

        # Analog input from voltage / current monitors
        data_in = list(self.analog_input)
        voltage, current, ifpower = np.array(data_in[::3]), np.array(data_in[1::3]), np.array(data_in[2::3])

        if debug:
            print("Raw voltage range: {:.2f} to {:.2f} V".format(voltage.min(), voltage.max()))
            print("Raw current range: {:.2f} to {:.2f} V".format(current.min(), current.max()))

        # Calibrate to V / A
        voltage = (voltage - self.params['VMON']['OFFSET']) / self.params['VMON']['GAIN']
        current = (current - self.params['IMON']['OFFSET']) / self.params['IMON']['GAIN']

        return voltage, current, ifpower

    def _read_analog(self, channel):
        """Read analog input channel.

        Args:
            channel: channel number

        Returns:
            analog input

        """

        return self.ai_device.a_in(channel, AI_MODE, AI_RANGE, AI_FLAG)

    def start_iv_monitor_scan(self, npts=1001, sweep_period=5.0, verbose=True):
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
        rate = self.ai_device.a_in_scan(self.params['VMON_AI_CHANNEL'], 
                                        self.params['PIF_AI_CHANNEL'],
                                        AI_MODE, AI_RANGE, samples_per_channel,
                                        sample_frequency, SCAN_OPTIONS,
                                        AI_SCAN_FLAG, self.analog_input)

        if verbose:
            print("\n\tReading voltage & current monitors:")
            print(f'\t\t{self.daq_name}: ready')
            print(f'\t\tRange: {AI_RANGE.name}')
            print(f'\t\tSweep period: {sweep_period:.1f} s')
            print(f'\t\tSweep frequency: {sample_frequency:.1f} Hz')
            print(f'\t\tSamples per channel: {samples_per_channel:d}')
            print(f'\t\tSample Rate: {sample_frequency:.1f} Hz')
            print(f'\t\tActual sample rate: {rate:.1f} Hz\n')

    # Plot --------------------------------------------------------------- ###

    def plot(self, mode="once"):
        """Plot I-V curve."""

        fig, ax = plt.subplots()
        ax.set_xlabel("Voltage (mV)")
        ax.set_ylabel("Current (uA)")
        ax.set_title("SIS bias control")

        # Once
        if mode == "once":
            voltage, current, _ = self.read_iv_curve()
            ax.plot(voltage*1e3, current*1e6, 'ko', alpha=0.2, ms=1)
            plt.show()
            return

        # TODO: allow to run continuously

        # # plt.show(False)
        # plt.draw()

        # points = ax.plot([0], [0], 'ko', alpha=0.2, ms=1)[0]

        # for _ in range(10):
        #     npts = 500
        #     voltage, current = np.empty(npts), np.empty(npts)
        #     for i in range(npts):
        #         voltage[i] = bias.read_voltage()
        #         current[i] = bias.read_current()
        #         time.sleep(0.001)
        #     # plt.plot(voltage, current, 'ko', alpha=0.2, ms=1)
        #     points.set_data(voltage, current)
        #     fig.canvas.draw()
        #     plt.pause(0.1)    

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

    def stop(self):
        """Stop DAQ device and close all connections."""
        
        self.close()
    
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
