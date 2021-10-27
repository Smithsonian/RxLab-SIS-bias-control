import os


def progress_bar(iteration, total, prefix='', length=50):
    percent = "{:.1f}".format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = "X" * filledLength + '-' * (length - filledLength)
    print(f"\r{prefix} |{bar}| {percent}%", end="\r")
    if iteration == total: 
        print("")


def ask_filename(msg="\tFile name: "):
    print("")
    while True:
        fname = input(msg)
        if fname == "":
            return None
        elif os.path.isfile(fname + ".dat") or os.path.isfile(fname + ".mdat"):
            ans = input("\tFile already exists. Overwrite? [y/n] ")
            if ans.lower() == "y":
                return fname
        else:
            print("")
            return fname
