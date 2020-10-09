"""
a script for wrapping around my python scripts
"""
import os
import time
import sys
import argparse
import logging
import subprocess


def main(args=sys.argv[1:]):
    target_file = args[0]
    run_line = " ".join(args[1:])
    output = subprocess.check_output(
       "qsub -cwd run_script.sh %s" % run_line,
       stderr=subprocess.STDOUT,
       shell=True)
    print(output)

    while not os.path.exists(target_file):
        time.sleep(10)

if __name__ == "__main__":
    main(sys.argv[1:])

