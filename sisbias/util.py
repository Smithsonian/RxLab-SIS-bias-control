"""Miscellaneous utilities."""

import os


def progress_bar(iteration, total, prefix='', length=50):
    """Print progress bar.

    Args:
        iteration (int): iteration number
        total (int): total iterations
        prefix (str): message to print before progress bar, default is ''
        length (int): character length of progress bar, default is 50

    """

    percent = "{:.1f}".format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = "X" * filledLength + '-' * (length - filledLength)
    print(f"\r{prefix} |{bar}| {percent}%", end="\r")
    if iteration == total: 
        print("")


def ask_filename(msg="\tFile name (omit extension): "):
    """Ask user for file name.

    Args:
        msg (str): message to print to user

    Returns:
        str: file name

    """

    print("")
    while True:
        fname = input(msg)
        print("")
        if fname == "":
            return None
        elif os.path.isfile(fname + ".dat") or os.path.isfile(fname + ".mdat"):
            ans = input("\tFile already exists. Overwrite? [y/n] ")
            if ans.lower() == "y":
                return fname
        else:
            return fname


# MAIN -------------------------------------------------------------------- ##

if __name__ == "__main__":

    import time

    filename = ask_filename()
    if filename is None:
        print("\tNo filename given.\n")

    npts = 100
    for i in range(npts+1):
        progress_bar(i, npts, prefix="\tProgress: ")
        time.sleep(0.01)
    print("")
