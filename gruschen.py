import sys
import traceback
import unittest

from cutoff import CutoffTest
from metrics import MetricsTest
from mfcc import MFCCTest
import signal_comparison
from sound_file import SoundFileTest

# TODO:  implement tests for all packages


class Recording:
    def __init__(self, name, text, filename):
        self.name = name
        self.text = text
        self.filename = filename

    def __str__(self):
        raise NotImplementedError


def run_tests():
    unittest.main()


if __name__ == "__main__":
    try:
        path = ""
        if len(sys.argv) > 1:
            path = sys.argv[1]
        files = [Recording(s + str(i), s, path + s + str(i) + ".wav")
                 for i in range(1, 11)
                 for s in ["yes", "no", "red", "green", "yellow"]]
        distances = signal_comparison.compare_files(files, False)
    except Exception as e:
        traceback.print_exc(file=sys.stdout)
        print(repr(e))
