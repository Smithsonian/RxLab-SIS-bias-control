"""Main script for controlling the SIS bias via the MCC DAQ device."""

import time
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt 
from sisbias import SISBias


# Default parameters
param = dict(
    VMIN        = -2.0,
    VMAX        = 2.0,
    PERIOD      = 5.0,
    SAMPLE_RATE = 1000.0,
)


try:
    # Initialize bias control
    bias = SISBias()

    # Interactive plots
    plt.ion()

    while True:

        # Get command
        command = input(">> ")
        command = command.split()
        if not isinstance(command, list):
            command = [command,]
        command[0] = command[0].upper()

        # Process command
        # If parameter, set parameter
        if command[0] in param.keys():
            param[command[0]] = float(command[1])

        # SWEEP / START: Sweep control voltage
        elif command[0] == "SWEEP" or command[0] == "START" or command[0] == "S":
            bias.sweep_control_voltage(vmin=param['VMIN'],
                                       vmax=param['VMAX'],
                                       period=param['PERIOD'],
                                       sample_rate=param['SAMPLE_RATE'])
            bias.start_iv_monitor_scan(period=param['PERIOD'],
                                       sample_rate=param['SAMPLE_RATE'])

        # PULSE: Pulse control voltage
        elif command[0] == "PULSE":
            print("")
            bias.pulse_control_voltage(vmin=param['VMIN'],
                                       vmax=param['VMAX'],
                                       period=param['PERIOD'],
                                       sample_rate=param['SAMPLE_RATE'])
            bias.start_iv_monitor_scan(period=param['PERIOD'],
                                       sample_rate=param['SAMPLE_RATE'])

        # VSET: Set constant control voltage
        elif command[0] == "VSET":
            vctrl = float(command[1])
            print("\n\tSet control voltage: {:.2f} V\n".format(vctrl))
            bias.set_control_voltage(vctrl)

        # VMON: Read voltage monitor
        elif command[0] == "VMON":
            vmon_mv = bias.read_voltage()
            print("\n\tVoltage monitor: {:.2f} mV\n".format(vmon_mv))

        # IMON: Read current monitor
        elif command[0] == "IMON":
            imon_ua = bias.read_current()
            print("\n\tCurrent monitor: {:.1f} mA\n".format(imon_ua))

        # INFO: Print info
        elif command[0] == "INFO":
            print("")
            for key, value in param.items():
                sep = "\t" if len(key) > 8 else "\t\t"
                print(f"\t{key}{sep}{value}")
            print("")

        # STATUS: Print scan status
        elif command[0] == "STATUS":
            print(f"\n\tScanning: {bias.ao_scan_status() == 1}\n")

        # PLOT: Plot I-V curve
        elif command[0] == "PLOT" or command[0] == "P":
            if not bias.ao_scan_status():
                print("You haven't started scanning yet...")

            bias.plot(mode="once")

        # CLEAR: Clear all plots
        elif command[0] == "CLEAR" or command[0] == "C":
            plt.close("all")

        # HELP: Print help
        elif command[0] == "HELP" or command[0] == "H":

            # Commands
            print("\n\tAvailable commands:")
            print("\t\tHELP or H: Print help")
            print("\t\tSWEEP or START or S: Sweep control voltage (triangle wave)")
            print("\t\tPULSE: Pulse control voltage (square wave)")
            print("\t\tVSET: Set constant control voltage")
            print("\t\tVMON: Read voltage monitor")
            print("\t\tIMON: Read current monitor")
            print("\t\tINFO: Print all parameters")
            print("\t\tSTATUS: Print scan status")
            print("\t\tPLOT or P: Plot I-V curve")
            print("\t\tCLEAR or C: Clear all plots")
            print("\t\tSTOP or EXIT or Q: Close connection")

            # Parameters
            print("\n\tAvailable parameters:")
            print("\t\tVMIN <value>: Minimum control voltage for sweep or pulse, in [V]")
            print("\t\tVMAX <value>: Maximum control voltage for sweep or pulse, in [V]")
            print("\t\tPERIOD <value>: Period of sweep or pulse, in [s]")
            print("\t\tSAMPLE_RATE <value>: Sample rate for control voltage sweep or pulse, in [Hz]")
            print("")

        # STOP: Stop bias and shutdown
        elif command[0] == "STOP" or command[0] == "EXIT" or command[0] == "Q":
            break

        # Command not recognized
        else:
            print("\n\tCommand not recognized.\n")

except KeyboardInterrupt:
    print("\nClosing program.")

except EOFError:
    print("\nClosing program.")

finally:
    bias.close()
