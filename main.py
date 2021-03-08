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
        # Set parameter
        if command[0] in param.keys():
            param[command[0]] = float(command[1])

        # Start voltage sweep
        elif command[0] == "SWEEP" or command[0] == "START":
            print("")
            bias.sweep_control_voltage(vmin=param['VMIN'],
                                       vmax=param['VMAX'],
                                       period=param['PERIOD'],
                                       sample_rate=param['SAMPLE_RATE'])

        elif command[0] == "PULSE":
            print("")
            bias.pulse_control_voltage(vmin=param['VMIN'],
                                       vmax=param['VMAX'],
                                       period=param['PERIOD'],
                                       sample_rate=param['SAMPLE_RATE'])

        # Print info to terminal
        elif command[0] == "INFO":
            for key, value in param.items():
                sep = "\t" if len(key) > 8 else "\t\t"
                print(f"\t{key}{sep}{value}")

        # Scan status
        elif command[0] == "STATUS":
            print(f"Scanning: {bias.ao_scan_status() == 1}")

        # Plot voltage and current
        elif command[0] == "PLOT":
            if not bias.ao_scan_status():
                print("You haven't started scanning yet...")

            fig, ax = plt.subplots()
            ax.set_xlabel("Voltage (mV)")
            ax.set_ylabel("Current (uA)")
            ax.set_title("SIS bias control")

            # Once
            npts = 5000
            voltage, current = np.empty(npts), np.empty(npts)
            for i in range(npts):
                voltage[i] = bias.read_voltage()
                current[i] = bias.read_current()
            ax.plot(voltage, current, 'ko', alpha=0.2, ms=1)
            plt.show()

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

        # Close all plots
        elif command[0] == "CLEAR":
            plt.close("all")

        # Stop
        elif command[0] == "STOP" or command[0] == "EXIT" or command[0] == "Q":
            break

        # Not recognized
        else:
            print("Command not recognized.")

except KeyboardInterrupt:
    print("\nClosing program.")

except EOFError:
    print("\nClosing program.")

finally:
    bias.close()
