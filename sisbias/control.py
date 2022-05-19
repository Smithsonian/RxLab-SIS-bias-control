"""Control the SIS bias board using an MCC DAQ device."""

import json
import time

import matplotlib.pyplot as plt
import numpy as np
import uldaq
from appdirs import user_config_dir
from scipy.optimize import minimize
from uldaq import (AiInputMode, AInFlag, AInScanFlag, AOutFlag, AOutScanFlag, DaqDevice, DigitalDirection,
                   InterfaceType, Range, ScanOption, ScanStatus, create_float_buffer,
                   get_daq_device_inventory)

from sisbias.filters import gauss_conv
from sisbias.util import progress_bar

# DAQ output channels
AO_FLAG = AOutFlag.DEFAULT
SCAN_OPTIONS = ScanOption.CONTINUOUS
SCAN_FLAGS = AOutScanFlag.DEFAULT

# DAQ input Channels
AI_MODE = AiInputMode.DIFFERENTIAL
AI_RANGE = Range.BIP5VOLTS
AI_FLAG = AInFlag.DEFAULT
AI_SCAN_FLAG = AInScanFlag.DEFAULT


class SISBias:
    """Class for controlling the SIS bias board using an MCC DAQ device.

    Args:
        config_file (str): configuration file, if None use default file,
            default is None
        cal_file (str): calibration file, if None use default file, default
            is None

    """
    
    def __init__(self, config_file=None, cal_file=None, daq_id=None, name=None, interface='usb'):
        
        self.name = name
        self.name_str = f"({self.name})"
        print(f"\nSIS BIAS CONTROL {self.name_str}")

        # Read configuration file
        self.config_file = config_file
        if self.config_file is None:
            self.config_file = user_config_dir("rxlab-sis-bias.config")
        try:
            with open(self.config_file) as _fin:
                self.config = json.load(_fin)
        except FileNotFoundError as e:
            print("\nConfiguration file not found.")
            print("Run \"sisbias-init-config-v0\" or \"sisbias-init-config-v3\" to initialize this file.\n")
            raise e

        # Read calibration file file
        self.cal_file = cal_file
        if self.cal_file is None:
            self.cal_file = user_config_dir("rxlab-sis-bias.cal")
        try:
            with open(self.cal_file) as _fin:
                self.cal = json.load(_fin)
        except FileNotFoundError as e:
            print("\nCalibration file not found.")
            print("Run \"sisbias-init-cal\" to initialize this file.\n")
            raise e

        # Find all available DAQ devices
        if interface.lower() == 'usb':
            self.ao_range = Range.UNI5VOLTS  # TODO: Fix hack
            devices = get_daq_device_inventory(InterfaceType.USB)
        elif interface.lower() == 'ethernet':
            self.ao_range = Range.BIP10VOLTS  # TODO: Fix hack
            devices = get_daq_device_inventory(InterfaceType.ETHERNET)
        else:
            print("interface must be either 'usb' or 'ethernet'")
            raise ValueError
        number_of_devices = len(devices)

        if daq_id is None:
            # Verify at least one DAQ device is detected
            if number_of_devices == 0:
                raise RuntimeError('Error: No DAQ devices found')
                
            # Print all available DAQ devices
            print(f'\nFound {number_of_devices} DAQ device(s):')
            for i in range(number_of_devices):
                print(f'  [{i}] {devices[i].product_name} ({devices[i].unique_id})')
                    
            # Choose DAQ device
            if number_of_devices == 1:
                descriptor_index = 0
            else:
                msg = f'\nPlease select a DAQ device (between 0 and {number_of_devices-1}): '
                descriptor_index = input(msg)
                descriptor_index = int(descriptor_index)
                if descriptor_index not in range(number_of_devices):
                    raise RuntimeError('Error: Invalid descriptor index')
        else:
            id_list = [item.unique_id for item in devices]
            try:
                descriptor_index = id_list.index(daq_id)
            except ValueError:
                print(f"\n\t**Specified DAQ ({daq_id}) not found.**\n")
                raise DAQNotFoundError
                # raise e
            
        # Create the DAQ device object
        self.daq_device = DaqDevice(devices[descriptor_index])
        self.ao_device = self.daq_device.get_ao_device()
        self.ai_device = self.daq_device.get_ai_device()
        self.dio_device = self.daq_device.get_dio_device()

        # Verify the specified DAQ device supports analog input
        if self.ai_device is None:
            raise RuntimeError('\nError: The DAQ device does not support analog input')
                               
        # Verify the specified DAQ device supports analog output
        if self.ao_device is None:
            raise RuntimeError('\nError: The DAQ device does not support analog output')

        # Verify the specified DAQ device supports digital input/output
        if self.dio_device is None:
            raise RuntimeError('\nError: The DAQ device does not support digital input/output')
                
        # Verify the device supports hardware pacing for analog output
        self.ao_info = self.ao_device.get_info()
        self.has_ao_pacer = self.ao_info.has_pacer()
        # print(self.ao_info.get_ranges())
        # TODO: add ability to specify AO range
        # TODO: add ability to specify AI range (per channel??)
        if not self.has_ao_pacer:
            print('\nWarning: This DAQ device does not support hardware-paced analog output')
                               
        # Establish a connection to the device.
        self.desc = self.daq_device.get_descriptor()
        self.daq_name = "{} ({})".format(self.desc.dev_string, self.desc.unique_id)
        print(f'\nConnecting to {self.daq_name}... ', end=" ")
        self.daq_device.connect(connection_code=0)
        print("done\n")
        
        # Initialize scan status
        if self.has_ao_pacer:
            self._ao_scan_status, self._ao_transfer_status = self.ao_device.get_scan_status()
        else:
            self._ao_scan_status, self._ao_transfer_status = False, False
        self._ai_scan_status, self._ai_transfer_status = self.ai_device.get_scan_status()
        self.analog_input = None

        # Initialize bit for controlling ambient load
        self.dio_info = self.dio_device.get_info()
        self.digital_port_type = self.dio_info.get_port_types()[0]
        self.dio_device.d_config_port(self.digital_port_type, DigitalDirection.OUTPUT)
        self.dio_device.d_bit_out(self.digital_port_type, 0, 0)

    def __repr__(self):

        return f"SIS bias control via MCC DAQ device: {self.daq_name}"

    def __str__(self):

        return self.__repr__()

    # Set control voltage ------------------------------------------------ ###

    def set_control_voltage(self, voltage, vlimit=5, verbose=False):
        """Set control voltage to a constant value (no sweep).

        Args:
            voltage (float): control voltage, in [V]
            vlimit (float): hard limit on control voltage, in [V]
            verbose (bool): print to terminal

        """

        # Stop current scan
        if self.has_ao_pacer:
            self.update_ao_scan_status()
        if self._ao_scan_status == ScanStatus.RUNNING:
            self.ao_device.scan_stop()

        # Check if within limits
        if voltage > vlimit:
            print(f"\n\t** warning: control voltage beyond limits ({-vlimit:.1f} to {vlimit:.1f} V) **")
            print(f"\tSetting control voltage to {vlimit:.1f} V\n")
            voltage = vlimit
        elif voltage < -vlimit:
            print(f"\n\t** warning: control voltage beyond limits ({-vlimit:.1f} to {vlimit:.1f} V) **")
            print(f"\tSetting control voltage to {-vlimit:.1f} V\n")
            voltage = -vlimit

        # Use two unipolar outputs to create differential output
        if voltage >= 0:
            self.ao_device.a_out(self.config['VCTRL']['AO_N_CHANNEL'], self.ao_range, AO_FLAG, 0.0)
            self.ao_device.a_out(self.config['VCTRL']['AO_P_CHANNEL'], self.ao_range, AO_FLAG, voltage)
        else:
            self.ao_device.a_out(self.config['VCTRL']['AO_N_CHANNEL'], self.ao_range, AO_FLAG, -voltage)
            self.ao_device.a_out(self.config['VCTRL']['AO_P_CHANNEL'], self.ao_range, AO_FLAG, 0.0)

        if verbose:
            print(f"Control voltage set to {voltage:.1f} V")
    
    def set_bias_voltage(self, vbias=0, dv=0.02, vstart=None, sleep_time=0.1, iterations=3, vlimit=5, verbose=False):
        """Set bias voltage to desired value.

        Uses Newton's method to hone in on the target value.

        Args:
            vbias (float): target bias voltage, in [mV], default is 0
            dv (float): voltage step for derivative, in [V], default is 0.02
            vstart (float): starting control voltage, in [V], default is None
            sleep_time (float): sleep time between voltage changes, in [s], default is 0.1
            iterations (int): iterations, default is 3
            vlimit (float): hard limit for control voltage, in [V], default is 5
            verbose (bool): print to terminal, default is False

        """

        # Starting value
        if vstart is not None:
            vctrl = vstart
        else:
            vctrl = self.bias_to_control_voltage(vbias)
        self.set_control_voltage(vctrl, vlimit=vlimit)
        time.sleep(sleep_time)

        # Read bias voltage
        vbias1 = self.read_voltage(average=100)
        if verbose:
            print("\tBias voltage:  {:6.2f} mV".format(vbias1))

        if iterations is None or iterations == 0:
            return 

        # Iterate to find control voltage
        for _ in range(iterations):
            
            # Calculate derivative
            self.set_control_voltage(vctrl + dv, vlimit=vlimit)
            time.sleep(sleep_time)
            vbias2 = self.read_voltage(average=100)
            der = (vbias2 - vbias1) / dv

            # Update control voltage
            error = vbias1 - vbias
            vctrl -= error / der 
            self.set_control_voltage(vctrl, vlimit=vlimit)
            time.sleep(sleep_time)

            # Read bias voltage
            vbias1 = self.read_voltage(average=100)
            if verbose:
                print("\tBias voltage:  {:6.2f} mV".format(vbias1))
        print("")
        
        return vbias1, vctrl

    def sweep_control_voltage(self, vmin=-1, vmax=1, npts=1000, sweep_period=5, vlimit=5, verbose=True):
        """Sweep control voltage (triangle wave).

        Uses two channels to create a differential output (to allow negative
        control voltages).

        Args:
            vmin (float): minimum control voltage, in [V], default is -1
            vmax (float): maximum control voltage, in [V], default is 1
            npts (int): number of points, default is 1000
            sweep_period (float): sweep period, in [s], default is 5.0
            vlimit (float): hard limit for control voltage, in [V], default is 5
            verbose (bool): print to terminal, default is False

        """

        # ensure that this DAQ support hardware pacing
        if not self.has_ao_pacer:
            print("\nWarning: this DAQ does not support hardware pacing.")
            print("You cannot use sweep_control_voltage with this DAQ")
            return

        # make sure vmin/vmax are within limits
        vlimit = abs(vlimit)
        msg = f"\n\t** warning: vmin or vmax is outside the limits ({-vlimit:.1f} to {vlimit:.1f} V) **"
        if vmax > vlimit:
            print(msg)
            vmax = abs(vlimit)
        if vmin > vlimit:
            print(msg)
            vmin = abs(vlimit)
        if vmax < -vlimit:
            print(msg)
            vmax = -vlimit
        if vmin < -vlimit:
            print(msg)
            vmin = -vlimit

        sample_period = sweep_period / npts
        sample_frequency = 1 / sample_period
        samples_per_channel = int(npts)

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
        self._sweep(vctrl_weave, sweep_period=sweep_period, sample_frequency=sample_frequency, verbose=verbose)

    def _sweep(self, voltage, sweep_period=5, sample_frequency=1000, verbose=True):
        """Sweep control voltage.

        Args:
            voltage (np.ndarray): voltage array for buffer
            sweep_period (float): period
            sample_frequency (float): sample frequency

        """

        samples_per_period = int(sample_frequency * sweep_period)
        samples_per_channel = int(samples_per_period)

        # Stop current scan
        if self.has_ao_pacer:
            self.update_ao_scan_status()
        if self._ao_scan_status == ScanStatus.RUNNING:
            self.ao_device.scan_stop()

        # Create output buffer for control voltage
        output_buffer = create_float_buffer(2, samples_per_channel)

        # Min / max value
        control_signal = voltage[1::2] - voltage[::2]
        vmin = control_signal.min()
        vmax = control_signal.max()
        npts = len(control_signal)

        # Fill buffer with data
        for i in range(len(voltage)):
            output_buffer[i] = voltage[i]

        # Start the output scan.
        vctrl_n_chan = self.config['VCTRL']['AO_N_CHANNEL']
        vctrl_p_chan = self.config['VCTRL']['AO_P_CHANNEL']
        rate = self.ao_device.a_out_scan(vctrl_n_chan, vctrl_p_chan,
                                         self.ao_range, samples_per_channel, 
                                         sample_frequency, SCAN_OPTIONS, 
                                         SCAN_FLAGS, output_buffer)

        if verbose:
            vbmin = self.control_to_bias_voltage(vmin)
            vbmax = self.control_to_bias_voltage(vmax)
            print("\n\tSweep control voltage:")
            print(f'\t\t{self.daq_name}: ready')
            print(f'\t\tDAC range:             {self.ao_range.name}')
            print(f'\t\tControl voltage range: {vmin:.1f} to {vmax:.1f} V')
            print(f'\t\tBias voltage range:    {vbmin:.1f} to {vbmax:.1f} mV')
            print(f'\t\tVoltage points:        {npts}')
            print(f'\t\tSweep frequency:       {1 / sweep_period:.1f} Hz')
            print(f'\t\tSweep period:          {sweep_period:.1f} s')
            print(f'\t\tSampling frequency:    {sample_frequency:.1f} Hz')
            print(f'\t\tSampling frequency:    {rate:.1f} Hz (actual)')

    def control_to_bias_voltage(self, vctrl):
        """Convert control voltage to bias voltage.

        I.e., the voltage coming from the DAQ.

        Args:
            vctrl (float): control voltage, in [V]

        Returns:
            float: bias voltage, in [mV]

        """

        return vctrl * 1000 / self.config['VMON']['GAIN']

    def bias_to_control_voltage(self, vbias):
        """Convert bias voltage to control voltage.

        I.e., the voltage across the SIS junction.

        Args:
            vbias (float): bias voltage, in [mV]

        Returns:
            float: control voltage, in [V]

        """

        return vbias / 1000 * self.config['VMON']['GAIN']

    # Read voltage & current monitor ------------------------------------- ###

    def read_voltage(self, average=1000, raw=False, calibrate=True, stats=False, verbose=False, msg=None):
        """Read voltage monitor.

        Returns:
            float: voltage monitor in [mV]

        """
        
        # Stop current scan
        self.update_ai_scan_status()
        if self._ai_scan_status == ScanStatus.RUNNING:
            self.ai_device.scan_stop()

        # Sample voltage monitor
        vmon = np.zeros(average)
        for i in range(average):
            vmon[i] = self._read_analog(self.config['VMON']['AI_CHANNEL'])

        # Convert from raw voltage to [mV]
        if not raw:
            vmon = vmon / self.config['VMON']['GAIN'] * 1e3  # mV

        # Calibrate
        if calibrate and not raw:
            vmon -= self.cal['VOFFSET']

        vmon_avg, vmon_std = np.mean(vmon), np.std(vmon)
        if verbose:
            if msg is None:
                msg = "\n\tVoltage monitor"
            print(f"{msg}: {vmon_avg:.3f} +/- {vmon_std:.3f} mV\n")

        if stats:
            return vmon_avg, vmon_std
        else:
            return vmon_avg

    def read_current(self, average=1000, raw=False, calibrate=True, stats=False, verbose=False, msg=None):
        """Read current monitor.

        Returns:
            current monitor in [uA]

        """
        
        # Stop current scan
        self.update_ai_scan_status()
        if self._ai_scan_status == ScanStatus.RUNNING:
            self.ai_device.scan_stop()

        # Sample current monitor
        imon = np.zeros(average)
        for i in range(average):
            imon[i] = self._read_analog(self.config['IMON']['AI_CHANNEL'])

        # Convert from raw voltage to [uA]
        if not raw:
            imon = imon / self.config['IMON']['GAIN'] * 1e6  # uA

        # Calibrate
        if calibrate and not raw:
            imon -= self.cal['IOFFSET']

        imon_avg, imon_std = np.mean(imon), np.std(imon)
        if verbose:
            if msg is None:
                msg = "\n\tCurrent monitor"
            print(f"{msg}: {imon_avg:.3f} +/- {imon_std:.3f} uA\n")

        if stats:
            return imon_avg, imon_std
        else:
            return imon_avg
    
    def read_ifpower(self, average=1000, calibrate=True, offset_only=False, stats=False, verbose=False, msg=None):
        """Read IF power from power meter/detector.

        Returns:
            IF power

        """

        # Stop current scan
        self.update_ai_scan_status()
        if self._ai_scan_status == ScanStatus.RUNNING:
            self.ai_device.scan_stop()

        # Sample IF power
        pif_raw = np.zeros(average)
        for i in range(average):
            pif_raw[i] = self._read_analog(self.config['PIF']['AI_CHANNEL'])

        # Remove power offset
        pif_offset = pif_raw - self.cal['IFOFFSET']

        # Calibrate: [AU] -> [K]
        pif_cal = pif_offset * self.cal['IFCORR']
        
        # Choose which value to return
        if offset_only:
            pif = pif_offset
        elif calibrate:
            pif = pif_cal
        else:
            pif = pif_raw

        if verbose:
            if msg is None:
                print("\n\tIF power")
            else:
                print(msg)
            print(f"\t\t- raw:              {np.mean(pif_raw*10):8.3f} +/- {np.std(pif_raw*10):8.3f} AU (x10)")
            print(f"\t\t- offset corrected: {np.mean(pif_offset*10):8.3f} +/- {np.std(pif_offset*10):8.3f} AU (x10)")
            print(f"\t\t- calibrated:       {np.mean(pif_cal):8.3f} +/- {np.std(pif_cal):8.3f} K\n")

        if stats:
            return np.mean(pif), np.std(pif)
        else:
            return np.mean(pif)

    def read_all(self, average=1000, raw=False, calibrate=True, stats=False, verbose=False):

        # Stop current scan
        self.update_ai_scan_status()
        if self._ai_scan_status == ScanStatus.RUNNING:
            self.ai_device.scan_stop()

        # Sample voltage, current, IF power
        vmon = np.zeros(average)
        imon = np.zeros(average)
        pif = np.zeros(average)
        for i in range(average):
            vmon[i] = self._read_analog(self.config['VMON']['AI_CHANNEL'])
            imon[i] = self._read_analog(self.config['IMON']['AI_CHANNEL'])
            pif[i] = self._read_analog(self.config['PIF']['AI_CHANNEL'])

        # Convert from raw voltage to desired units
        if not raw:
            vmon = vmon / self.config['VMON']['GAIN'] * 1e3  # mV
            imon = imon / self.config['IMON']['GAIN'] * 1e6  # uA

        # Calibrate
        if calibrate and not raw:
            vmon -= self.cal['VOFFSET']
            imon -= self.cal['IOFFSET']
            pif = (pif - self.cal['IFOFFSET']) * self.cal['IFCORR']

        vmon_avg, vmon_std = np.mean(vmon), np.std(vmon)
        imon_avg, imon_std = np.mean(imon), np.std(imon)
        pif_avg, pif_std = np.mean(pif), np.std(pif)
        if verbose:
            print(f"\n\tVoltage monitor: {vmon_avg:6.3f} +/- {vmon_std:6.3f} mV\n")
            print(f"\n\tCurrent monitor: {imon_avg:6.3f} +/- {imon_std:6.3f} uA\n")
            print(f"\n\tIF power:        {pif_avg:6.3f} +/- {pif_std:6.3f} K\n")

        if stats:
            return vmon_avg, vmon_std, imon_avg, imon_std, pif_avg, pif_std
        else:
            return vmon_avg, imon_avg, pif_avg

    def read_iv_curve_buffer(self, raw=False, calibrate=True, debug=False):
        """Read I-V curve.

        Returns:
            voltage in V, current in A, IF power in A.U.

        """

        # Analog input from voltage / current monitors
        data_in = list(self.analog_input)
        vmon, imon, pif = np.array(data_in[::3]), np.array(data_in[1::3]), np.array(data_in[2::3])

        if debug:
            print(f"Raw voltage range: {vmon.min():.2f} to {vmon.max():.2f} V")
            print(f"Raw current range: {imon.min():.2f} to {imon.max():.2f} V")

        # Convert from raw voltage to desired units
        if not raw:
            vmon = vmon / self.config['VMON']['GAIN'] * 1e3  # mV
            imon = imon / self.config['IMON']['GAIN'] * 1e6  # uA

        # Calibrate
        if calibrate and not raw:
            vmon -= self.cal['VOFFSET']
            imon -= self.cal['IOFFSET']
            pif = (pif - self.cal['IFOFFSET']) * self.cal['IFCORR']

        return vmon, imon, pif

    def _read_analog(self, channel):
        """Read analog input channel.

        Args:
            channel (int): channel number

        Returns:
            np.ndarray: analog input

        """

        return self.ai_device.a_in(channel, AI_MODE, AI_RANGE, AI_FLAG)

    def start_iv_monitor_scan(self, npts=1000, sweep_period=5, verbose=True):
        """Scan voltage & current monitors.

        Args:
            npts (int): number of points in IV scan, default is 1000
            sweep_period (float): sweep period in seconds, default is 5 seconds
            verbose (bool): print info to terminal?, default is True

        """

        sample_period = sweep_period / npts
        sample_frequency = 1 / sample_period
        samples_per_channel = int(npts)

        # Stop current scan
        self.update_ai_scan_status()
        if self._ai_scan_status == ScanStatus.RUNNING:
            self.ai_device.scan_stop()

        # Allocate a buffer to receive the data.
        self.analog_input = create_float_buffer(3, samples_per_channel)

        # Start the acquisition.
        rate = self.ai_device.a_in_scan(self.config['VMON']['AI_CHANNEL'], 
                                        self.config['PIF']['AI_CHANNEL'],
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

    # Calibrate ---------------------------------------------------------- ###

    def calibrate_iv_offset(self, npts=101, average=64, vmin=-0.5, vmax=0.5, verbose=True, debug=False):
        """Calibrate offset in I-V curve.

        Args:
            npts (int): number of data points, default is 51
            average (int): averaging, default is 64
            vmin (float): minimum control voltage, default is -0.5
            vmax (float): maximum control voltage, default is 0.5
            verbose (bool): verbosity, default is True
            debug (bool): debugging, plots data, default is False

        Returns:
            tuple: voltage and current offset
        """

        # Measure I-V curve
        try:
            data = self.measure_ivif(npts=npts, average=average, vmin=vmin,
                                     vmax=vmax, vlimit=5, sleep_time=0.1,
                                     stats=False, calibrate=False, verbose=True)
        except KeyboardInterrupt:
            print("")
            plt.close('all')
            return
        voltage, current = data[0, :], data[1, :]

        # Offset model
        def model(offset):
            # I-V curve
            vv1, ii1 = voltage - offset[0], current - offset[1]
            # Flipped I-V curve
            vv2, ii2 = -vv1[::-1], -ii1[::-1]
            # Interpolate to common voltage
            xx = np.linspace(-2, 2, 101)
            yy1 = np.interp(xx, vv1, ii1)
            yy2 = np.interp(xx, vv2, ii2)
            # Find total error
            return np.sum(np.abs(yy1 - yy2))

        # Minimize to find I-V offsets
        x0 = np.array([0, 0])
        res = minimize(model, x0=x0)
        voffset = res.x[0]
        ioffset = res.x[1]

        # Force i=0 v=0
        vtmp = np.linspace(-0.1, 0.1, 1001)
        itmp = np.interp(vtmp, data[0, :] - voffset, data[1, :] - ioffset)
        voffset += vtmp[np.abs(itmp).argmin()]

        # Print to terminal
        if verbose:
            print(f"\n\tVoltage offset: {self.cal['VOFFSET']:7.4f} mV (previous)")
            print(f"  \tVoltage offset: {voffset:7.4f} mV (new)")
            print(f"\n\tCurrent offset: {self.cal['IOFFSET']:7.4f} uA (previous)")
            print(f"  \tCurrent offset: {ioffset:7.4f} uA (new)\n")

        # Save to attributes
        self.cal['VOFFSET'] = voffset
        self.cal['IOFFSET'] = ioffset

        if debug:
            plt.figure(figsize=(6, 5))
            plt.title(self.name_str[1:-1])
            # I-V curve
            v1, i1 = voltage - voffset, current - ioffset
            plt.plot(v1, i1, 'k', label="I-V")
            # Flipped I-V curve
            v2, i2 = -v1[::-1], -i1[::-1]
            plt.plot(v2, i2, 'r', label="Flipped I-V")
            plt.axvline(0, c='k', lw=0.5)
            plt.axhline(0, c='k', lw=0.5)
            plt.xlabel("Bias voltage (mV)")
            plt.ylabel("Current (uA)")
            plt.legend()
            plt.tight_layout()
            plt.show()

        return self.cal['VOFFSET'], self.cal['IOFFSET']

    def calibrate_if_offset(self, vcontrol=0.2, average=10_000, verbose=True, wait1=True, wait2=True):
        """Calibrate IF power offset.

        Args:
            vcontrol (float): control voltage for offset integration, default
                is 0.2
            average (int): averaging, default is 10000
            verbose (bool): verbosity, default is True
            wait1 (bool): wait to turn off LNA?, default is True
            wait2 (bool): wait to turn back on LNA?, default is True

        Returns:
            float: if power offset, in units [AU]

        """

        # Calculate IF power offset
        if wait1:
            _ = input("\n\t\t** Turn warm LNA off. Press enter when ready. **")
        self.set_control_voltage(-vcontrol)
        time.sleep(0.2)
        if_offset1 = self.read_ifpower(average=average, calibrate=False)
        self.set_control_voltage(vcontrol)
        time.sleep(0.2)
        if_offset2 = self.read_ifpower(average=average, calibrate=False)
        new_ifoffset = (if_offset1 + if_offset2) / 2
        old_ifoffset = self.cal['IFOFFSET']
        self.cal['IFOFFSET'] = new_ifoffset
        if verbose:
            change = abs(old_ifoffset - new_ifoffset) / abs(new_ifoffset) * 100
            print(f"\n\tOld IF offset:  {old_ifoffset:7.4f} AU")
            print(f"  \tNew IF offset:  {new_ifoffset:7.4f} AU")
            print(f"  \t-> change:      {change:7.1f} %\n")
        if wait2:
            _ = input("\t\t** Turn warm LNA back on. Press enter when ready. **")

        return self.cal['IFOFFSET']

    def calibrate_if(self, vmin=1, vmax=1.5, average=2000, npts=10, sleep_time=0.2, njunc=3, extra=False, verbose=True,
                     debug=False, bias2=None):
        """Calibrate IF power using shot noise slope.

        Args:
            vmin (float): minimum control voltage, in [V], default is 1.0
            vmax (float): maximum control voltage, in [V], default is 1.5
            average (int): averaging, default is 1000
            npts (int): number of points, default is 10
            sleep_time (float): time to sleep between points, default is 0.2
            njunc (int): number of junctions, default is 3
            extra (bool): return extra info
            verbose (bool): verbosity, default is True
            debug (bool): debug, default is False
            bias2 (sisbias.SISBias): analyze second bias system at the same time, default is None

        Returns:
            IF calibration, in [K/AU]

        """

        # Initialize some variables
        vsis2, pif2 = None, None
        pshot2, au2k2 = None, None

        # Measure shot noise
        vsweep = np.linspace(vmin, vmax, npts)
        vsis, pif = np.zeros(npts), np.zeros(npts)
        if bias2:
            vsis2, pif2 = np.zeros(npts), np.zeros(npts)
        for i, _v in np.ndenumerate(vsweep):
            progress_bar(i[0]+1, npts, prefix="\tMeasuring shot noise: ")
            self.set_control_voltage(_v)
            if bias2:
                bias2.set_control_voltage(_v)
            time.sleep(sleep_time)
            vsis[i] = self.read_voltage(average=average)
            pif[i] = self.read_ifpower(average=average, calibrate=False)
            if bias2:
                vsis2[i] = bias2.read_voltage(average=average)
                pif2[i] = bias2.read_ifpower(average=average, calibrate=False)
        pif -= self.cal['IFOFFSET']
        if bias2:
            pif -= bias2.cal['IFOFFSET']

        # Calculate calibration factor
        pshot = np.polyfit(vsis, pif, 1)
        au2k = 5.8 / pshot[0] / njunc
        old_correction = self.cal['IFCORR']
        self.cal['IFCORR'] = au2k
        pif *= au2k
        change = abs(old_correction - au2k) / abs(au2k) * 100
        if verbose:
            if bias2:
                print("\n\tChannel 1:")
            print(f"\n\tOld correction: {old_correction:8.1f} K/AU")
            print(f"  \tNew correction: {self.cal['IFCORR']:8.1f} K/AU")
            print(f"  \t-> change:      {change:8.1f} %\n")

        if bias2:
            # Calculate calibration factor
            pshot2 = np.polyfit(vsis2, pif2, 1)
            au2k2 = 5.8 / pshot2[0] / njunc
            old_correction = bias2.cal['IFCORR']
            bias2.cal['IFCORR'] = au2k2
            pif2 *= au2k2
            change = abs(old_correction - au2k2) / abs(au2k2) * 100
            if verbose:
                print("\n\tChannel 2:")
                print(f"\n\tOld correction: {old_correction:8.1f} K/AU")
                print(f"  \tNew correction: {bias2.cal['IFCORR']:8.1f} K/AU")
                print(f"  \t-> change:      {change:8.1f} %\n")

        if debug:
            fig, ax1 = plt.subplots(figsize=(6, 5))
            plt.title(self.name_str[1:-1])
            ax1.plot(vsis, pif, 'ko-', label="IF power")
            ax1.plot(vsis, np.polyval(pshot, vsis) * au2k, 'r--')
            if bias2:
                ax1.plot(vsis2, pif2, 'bo-', label="IF power 2")
                ax1.plot(vsis2, np.polyval(pshot2, vsis2) * au2k2, 'r--')
            ax1.set_xlabel("Bias voltage (mV)")
            ax1.set_ylabel("IF power (AU)")
            ax1.legend()
            plt.show()

        if extra:
            if bias2 is None:
                return self.cal['IFCORR'], vsis, pif
            else:
                return self.cal['IFCORR'], vsis, pif, bias2.cal['IFCORR'], vsis2, pif2
        else:
            if bias2 is None:
                return self.cal['IFCORR']
            else:
                return self.cal['IFCORR'], bias2.cal['IFCORR']

    # Measure I-V/IF data ------------------------------------------------ ###

    def measure_ivif(self, npts=201, average=64, vmin=-1, vmax=1, vlimit=5, sleep_time=0.1, stats=True, msg=None,
                     calibrate=True, verbose=True):
        """Measure I-V curve and IF power as a function of bias voltage.

        Args:
            npts (int): number of points, default is 201
            average (int): averaging, default is 64
            vmin (float): minimum control voltage, in [V], default is -1
            vmax (float): maximum control voltage, in [V], default is -1
            vlimit (float): hard limit on control voltage, in [V], default is 1
            sleep_time (float): sleep time between voltage points, in [s], default is 0.1
            stats (bool): return statistics, default is True
            msg (str): progress message
            calibrate (bool): calibrate data, default is True
            verbose (bool): verbosity, default is True

        Returns:
            tuple: voltage, voltage error, current, current error, IF power, IF power error

        """

        if msg is None:
            msg = "\tProgress: "

        if stats:
            size = 6 
        else:
            size = 3

        # parameters for reading data
        _param = dict(average=average, stats=stats, calibrate=calibrate)

        vctrl_sweep = np.linspace(vmin, vmax, npts)
        try:
            results = np.zeros((size, npts))
            for i, _vctrl in np.ndenumerate(vctrl_sweep):
                self.set_control_voltage(_vctrl, vlimit=vlimit)
                time.sleep(sleep_time)
                results[:, i] = np.array(self.read_all(**_param)).reshape(size, 1)
                if verbose:
                    progress_bar(i[0] + 1, len(vctrl_sweep), prefix=msg)
        except KeyboardInterrupt:
            print("")
            plt.close('all')
            return

        # Sort by voltage
        idx = results[0, :].argsort()
        
        return results[:, idx]

    def plot_ivif(self, npts=201, average=64, vmin=-1, vmax=1, vlimit=5, sleep_time=0.1, msg=None, verbose=True):
        """Plot I-V curve and IF power as a function of bias voltage.

        Args:
            npts (int): number of points, default is 201
            average (int): averaging, default is 64
            vmin (float): minimum control voltage, in [V], default is -1
            vmax (float): maximum control voltage, in [V], default is -1
            vlimit (float): hard limit on control voltage, in [V], default is 1
            sleep_time (float): sleep time between voltage points, in [s], default is 0.1
            msg (str): message to print while measuring IV/IF data, default is None
            verbose (bool): verbosity, default is True

        Returns:
            tuple: voltage, voltage error, current, current error, IF power, IF power error

        """

        print("")

        results = self.measure_ivif(npts=npts, average=average, vmin=vmin, vmax=vmax, vlimit=vlimit, 
                                    sleep_time=sleep_time, calibrate=True, stats=False, msg=msg, verbose=verbose)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(results[0], results[1])
        ax2.plot(results[0], results[2])
        ax1.set_xlabel("Voltage (mV)")
        ax2.set_xlabel("Voltage (mV)")
        ax1.set_ylabel("Current (uA)")
        ax2.set_ylabel("IF power (K)")
        plt.show()

        return results

    def monitor(self, npts=1000, period=0.2, vmin=-1, vmax=1, vlimit=5, resistance=None, vctrl=0):
        """Plot real-time monitor.

        Args:
            npts (int): number of points, default is 1000
            period (float): period, in [s], default is 0.2
            vmin (float): minimum control voltage, in [V], default is -1
            vmax (float): maximum control voltage, in [V], default is 1
            vlimit (float): hard limit on control voltage, in [V], default is 5
            resistance (float): plot a line of constant resistance, in [ohms], default is None
            vctrl (float): control voltage to set upon completion, default is 0

        """

        # Start I-V bias sweeps
        self.sweep_control_voltage(vmin, vmax, npts=npts, sweep_period=period, vlimit=vlimit)
        self.start_iv_monitor_scan(npts=npts, sweep_period=period)
        time.sleep(period * 2)
        print("")

        # Read I-V curve
        voltage, current, ifpower = self.read_iv_curve_buffer()

        # Create figure
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.set_xlabel("Voltage (mV)")
        ax2.set_xlabel("Voltage (mV)")
        ax1.set_ylabel("Current (uA)")
        ax2.set_ylabel("IF power (K)")
        ax1.set_xlim([voltage.min(), voltage.max()])
        ax2.set_xlim([voltage.min(), voltage.max()])
        if resistance is not None:
            irmin = voltage.min() / resistance * 1e3
            irmax = voltage.max() / resistance * 1e3
            ax1.set_ylim([irmin, irmax])
            _, = ax1.plot([voltage.min(), voltage.max()], [irmin, irmax], 'r', label=f"{resistance:.0f} ohms")
        else:
            ax1.set_ylim([current.min(), current.max()])

        ax2.set_ylim([0, ifpower.max() * 2])
        line1, = ax1.plot([0], [0], 'k.', ms=1, label="Data")
        line2, = ax2.plot([0], [0], 'k.', ms=1, label="Data")
        if resistance is not None:
            ax1.legend()
        fig.canvas.draw()
        plt.show()

        while True:
            try:
                # Restart scans
                self.sweep_control_voltage(vmin, vmax, npts=npts, sweep_period=period, vlimit=vlimit, verbose=False)
                self.start_iv_monitor_scan(npts=npts, sweep_period=period, verbose=False)
                time.sleep(period)

                # Read I-V curve
                voltage, current, ifpower = self.read_iv_curve_buffer()

                # Draw I-V curve
                line1.set_data(voltage, current)
                line2.set_data(voltage, ifpower)
                fig.canvas.draw()
                plt.pause(0.0001)
                fig.canvas.flush_events()

            except KeyboardInterrupt:
                plt.close('all')
                self.set_control_voltage(vctrl)
                print(f"\n\tControl voltage returned to {vctrl:.2f} V.\n")
                break

    def noise_statistics(self, npts=50000, bias2=None):
        """Set control voltage to a constant value and then read-out monitors continuously.

        This method can help to measure fluctuations over time or interference.

        Args:
            npts (int): number of points, default is 50000
            bias2 (sisbias.SISBias): analyze a second bias system at the same time, default is None

        """

        # Read starting monitor values
        vmon = self.read_voltage(average=100)
        imon = self.read_current(average=100)
        if bias2:
            print("\tChannel 1:")
        print(f"\tVoltage monitor: {vmon:4.1f} mV")
        print(f"\tCurrent monitor: {imon:4.1f} uA\n")
        if bias2:
            print("\tChannel 2:")
            vmon = bias2.read_voltage(average=100)
            imon = bias2.read_current(average=100)
            print(f"\tVoltage monitor: {vmon:4.1f} mV")
            print(f"\tCurrent monitor: {imon:4.1f} uA\n")

        # Sample voltage/current/power monitors
        i = 0
        time_start = time.time()
        voltage1 = np.zeros(npts)
        current1 = np.zeros(npts)
        ifpower1 = np.zeros(npts)
        if bias2:
            voltage2 = np.zeros(npts)
            current2 = np.zeros(npts)
            ifpower2 = np.zeros(npts)
        else:
            voltage2, current2, ifpower2 = None, None, None
        try:
            # Read monitors
            for i in range(npts):
                voltage1[i] = self.read_voltage(average=1)
                current1[i] = self.read_current(average=1)
                ifpower1[i] = self.read_ifpower(average=1)
                if bias2:
                    voltage2[i] = bias2.read_voltage(average=1)
                    current2[i] = bias2.read_current(average=1)
                    ifpower2[i] = bias2.read_ifpower(average=1)

                if i % 1000 == 0 or i + 1 == npts:

                    # Use first 1000 samples to guess how long it will take
                    if i == 1000:
                        t_first_1000 = time.time() - time_start
                        t_guess = t_first_1000 * npts / 1000 
                        print(f"\tEst. meas. time: {t_guess:.0f} s / {t_guess/60:.0f} min / {t_guess/3600:.0f} hrs\n")

                    # Print progress bar with current pump level
                    if i > 0:
                        if bias2:
                            suffix = f' - {current1[i]:.0f} / {current2[i]:.0f} uA'
                        else:
                            suffix = f' - {current1[i]:.0f} uA'
                        progress_bar(i + 1, npts, prefix="\tProgress: ", suffix=suffix)

        except KeyboardInterrupt:
            pass

        # truncate data if the sweep is cut short
        if i + 1 < npts:
            voltage1 = voltage1[:i]
            current1 = current1[:i]
            ifpower1 = ifpower1[:i]
            if bias2:
                voltage2 = voltage2[:i]
                current2 = current2[:i]
                ifpower2 = ifpower2[:i]
            npts = i

        time_end = time.time()
        total_time = time_end - time_start
        # approx time samples (assuming even sampling)
        t = np.linspace(0, total_time, npts)
        # in Linux time
        time_out = np.linspace(time_start, time_end, npts)

        # assuming IF power range is 10uW
        ifpower1 *= 10
        if bias2:
            ifpower2 *= 10

        # Print statistics
        print("")
        if bias2:
            print("\tChannel 1:\n")
        print("\tVoltage monitor:\n")
        print("\t\tMean:               {:7.3f} mV".format(np.mean(voltage1)))
        print("\t\tStandard deviation: {:7.3f} mV".format(np.std(voltage1)))
        print("\t\t                    {:7.3f} % ".format(np.std(voltage1)/np.mean(voltage1)*100))
        print("\tCurrent monitor:")
        print("\t\tMean:               {:7.2f} uA".format(np.mean(current1)))
        print("\t\tStandard deviation: {:7.2f} uA".format(np.std(current1)))
        print("\t\t                    {:7.2f} % ".format(np.std(current1)/np.mean(current1)*100))
        print("\tIF power:")
        print("\t\tMean:               {:7.3f} K".format(np.mean(ifpower1)))
        print("\t\tStandard deviation: {:7.3f} K".format(np.std(ifpower1)))
        print("\t\t                    {:7.1f} % ".format(np.std(ifpower1)/np.mean(ifpower1)*100))
        if bias2:
            print("\n\tChannel 2:\n")
            print("\tVoltage monitor:")
            print("\t\tMean:               {:7.3f} mV".format(np.mean(voltage2)))
            print("\t\tStandard deviation: {:7.3f} mV".format(np.std(voltage2)))
            print("\t\t                    {:7.3f} % ".format(np.std(voltage2)/np.mean(voltage2)*100))
            print("\tCurrent monitor:")
            print("\t\tMean:               {:7.2f} uA".format(np.mean(current2)))
            print("\t\tStandard deviation: {:7.2f} uA".format(np.std(current2)))
            print("\t\t                    {:7.2f} % ".format(np.std(current2)/np.mean(current2)*100))
            print("\tIF power:")
            print("\t\tMean:               {:7.3f} K".format(np.mean(ifpower2)))
            print("\t\tStandard deviation: {:7.3f} K".format(np.std(ifpower2)))
            print("\t\t                    {:7.1f} % ".format(np.std(ifpower2)/np.mean(ifpower2)*100))
        print("\n\tSampling frequency: {:7.1f} kHz".format(npts/total_time/1e3))
        print("\n\tNumber of points:   {:7.0f}".format(npts))
        print("\n\tTotal time:         {:7.1f} s".format(total_time))
        print("\n\t                    {:7.1f} min\n".format(total_time / 60))

        # Plot
        fig, ((ax1, ax3, ax5), (ax2, ax4, ax6)) = plt.subplots(2, 3, figsize=(15, 8))
        ax1.plot(t, voltage1, 'k', lw=0.5, alpha=0.15)
        ax1.plot(t, gauss_conv(voltage1, 3), 'r', label="Channel 1")
        ax3.plot(t, current1, 'k', lw=0.5, alpha=0.15)
        ax3.plot(t, gauss_conv(current1, 3), 'r', label="Channel 1")
        ax5.plot(t, ifpower1, 'k', lw=0.5, alpha=0.15)
        ax5.plot(t, gauss_conv(ifpower1, 3), 'r', label="Channel 1")
        if bias2:
            ax1.plot(t, voltage2, 'k', lw=0.5, alpha=0.15)
            ax1.plot(t, gauss_conv(voltage2, 3), 'b', label="Channel 2")
            ax3.plot(t, current2, 'k', lw=0.5, alpha=0.15)
            ax3.plot(t, gauss_conv(current2, 3), 'b', label="Channel 2")
            ax5.plot(t, ifpower2, 'k', lw=0.5, alpha=0.15)
            ax5.plot(t, gauss_conv(ifpower2, 3), 'b', label="Channel 2")

        # FFT
        voltage1_fft = np.fft.fftshift(np.fft.fft(voltage1))
        current1_fft = np.fft.fftshift(np.fft.fft(current1))
        ifpower1_fft = np.fft.fftshift(np.fft.fft(ifpower1))
        f1 = np.fft.fftshift(np.fft.fftfreq(len(voltage1), d=t[1]-t[0]))
        if bias2:
            voltage2_fft = np.fft.fftshift(np.fft.fft(voltage2))
            current2_fft = np.fft.fftshift(np.fft.fft(current2))
            ifpower2_fft = np.fft.fftshift(np.fft.fft(ifpower2))
            f2 = np.fft.fftshift(np.fft.fftfreq(len(voltage2), d=t[1]-t[0]))
        else:
            f2 = None
            voltage2_fft, current2_fft, ifpower2_fft = None, None, None

        # Peak values
        if bias2:
            print("\tChannel 1:")
        idx = np.abs(voltage1_fft).argmax()
        print(f"\t\tPeak voltage:  {np.abs(voltage1_fft[idx]):12.1f} at {abs(f1[idx]):4.1f} Hz")
        idx = np.abs(current1_fft).argmax()
        print(f"\t\tPeak current:  {np.abs(current1_fft[idx]):12.1f} at {abs(f1[idx]):4.1f} Hz")
        idx = np.abs(ifpower1_fft).argmax()
        print(f"\t\tPeak IF power: {np.abs(ifpower1_fft[idx]):12.1f} at {abs(f1[idx]):4.1f} Hz\n")
        if bias2:
            print("\tChannel 2:")
            idx = np.abs(voltage2_fft).argmax()
            print(f"\t\tPeak voltage:  {np.abs(voltage2_fft[idx]):12.1f} at {abs(f2[idx]):4.1f} Hz")
            idx = np.abs(current2_fft).argmax()
            print(f"\t\tPeak current:  {np.abs(current2_fft[idx]):12.1f} at {abs(f2[idx]):4.1f} Hz")
            idx = np.abs(ifpower2_fft).argmax()
            print(f"\t\tPeak IF power: {np.abs(ifpower2_fft[idx]):12.1f} at {abs(f2[idx]):4.1f} Hz\n")

        # Plot
        ax2.loglog(f1, np.abs(voltage1_fft), 'k', lw=0.5, alpha=0.2)
        ax2.loglog(f1, gauss_conv(np.abs(voltage1_fft), 3), 'r', lw=2)
        ax2.axvspan(55, 65, color='r', alpha=0.2)
        ax4.loglog(f1, np.abs(current1_fft), 'k', lw=0.5, alpha=0.2)
        ax4.loglog(f1, gauss_conv(np.abs(current1_fft), 3), 'r', lw=2)
        ax4.axvspan(55, 65, color='r', alpha=0.2)
        ax6.loglog(f1, np.abs(ifpower1_fft), 'k', lw=0.5, alpha=0.5)
        ax6.loglog(f1, gauss_conv(np.abs(ifpower1_fft), 1), 'r', lw=2)
        ax6.axvspan(1.1, 1.3, color='r', alpha=0.2)
        if bias2:
            ax2.loglog(f2, np.abs(voltage2_fft), 'k', lw=0.5, alpha=0.2)
            ax2.loglog(f2, gauss_conv(np.abs(voltage2_fft), 3), 'b', lw=2)
            ax4.loglog(f2, np.abs(current2_fft), 'k', lw=0.5, alpha=0.2)
            ax4.loglog(f2, gauss_conv(np.abs(current2_fft), 3), 'b', lw=2)
            ax6.loglog(f2, np.abs(ifpower2_fft), 'k', lw=0.5, alpha=0.5)
            ax6.loglog(f2, gauss_conv(np.abs(ifpower2_fft), 1), 'b', lw=2)
        ax1.set_xlabel("Time (s)")
        ax3.set_xlabel("Time (s)")
        ax5.set_xlabel("Time (s)")
        ax1.set_ylabel("Voltage (mV)")
        ax3.set_ylabel("Current (uA)")
        ax5.set_ylabel("IF Power (K)")
        ax2.set_xlabel("Frequency (Hz)")
        ax4.set_xlabel("Frequency (Hz)")
        ax6.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Voltage")
        ax4.set_ylabel("Current")
        ax6.set_ylabel("IF Power")
        ax1.set_xlim([0, t.max()])
        ax3.set_xlim([0, t.max()])
        ax5.set_xlim([0, t.max()])
        ax2.set_xlim([1/total_time*2, npts/total_time/2])
        ax4.set_xlim([1/total_time*2, npts/total_time/2])
        ax6.set_xlim([1/total_time*2, npts/total_time/2])
        plt.show()

        if bias2:
            return np.vstack((time_out, voltage1, current1, ifpower1, voltage2, current2, ifpower2)).T
        else:
            return np.vstack((time_out, voltage1, current1, ifpower1)).T

    # Digital input / output --------------------------------------------- ###

    def hot_load(self):
        """Move ambient load to insert hot load."""

        self.dio_device.d_bit_out(self.digital_port_type, 0, 0)

    def cold_load(self):
        """Move ambient load to insert cold load."""

        self.dio_device.d_bit_out(self.digital_port_type, 0, 1)

    # Scan status -------------------------------------------------------- ###
    
    def ao_scan_status(self):
        """Update and return scan status."""

        if self.has_ao_pacer:
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

    # Save parameters ---------------------------------------------------- ###

    def save_cal(self, cal_filename=None):
        """Save calibration parameters.

        Args:
            cal_filename (str): calibration file name

        """

        if cal_filename is None:
            cal_filename = self.cal_file

        # Save parameters to file (for persistence)
        with open(cal_filename, 'w') as fout:
            json.dump(self.cal, fout, indent=4)
        print(f"\nCalibration parameters {self.name_str} saved to: {cal_filename}")

    def save_config(self, config_filename=None):
        """Save configuration parameters to file.

        Args:
            config_filename (str): configuration file name

        """

        if config_filename is None:
            config_filename = self.config_file

        # Save parameters to file (for persistence)
        with open(config_filename, 'w') as fout:
            json.dump(self.config, fout, indent=4)
        print(f"\nConfiguration parameters {self.name_str} saved to: {config_filename}")

    # Stop --------------------------------------------------------------- ###
    
    def close(self):
        """Stop DAQ device and close all connections."""

        print(f"Closing connection to DAQ device {self.name_str} ... ", end='')

        try:
            if self.daq_device:

                # Stop the scan
                if self.daq_device.is_connected() and self.has_ao_pacer:
                    self.update_ao_scan_status()
                if self.daq_device.is_connected() and self._ao_scan_status == ScanStatus.RUNNING:
                    self.ao_device.scan_stop()
                if self.daq_device.is_connected():
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


class DAQNotFoundError(Exception):
    """Error for when the DAQ can't be found."""
    pass

