import sys
import traceback
import unittest

from cutoff import CutoffTest
from mfcc import MFCCTest
import signal_comparison
from sound_file import SoundFileTest

# TODO:  implement tests for all packages


class Recording:
    def __init__(self, print_name, text, filename):
        self.__print_name = print_name
        self.text = text
        self.filename = filename

    def __str__(self):
        return self.__print_name


def run_tests():
    unittest.main()


if __name__ == "__main__":
    try:
        path = ""
        if len(sys.argv) > 1:
            path = sys.argv[1]
        files = {s + str(i): Recording(s + str(i),
                                       s,
                                       path + s + str(i) + ".wav")
                 for i in range(1, 15)
                 for s in ["yes", "no", "red", "green", "yellow"]}
        metrics = signal_comparison.compare_files(files, False)
    except Exception as e:
        traceback.print_exc(file=sys.stdout)
        print(repr(e))
