import sys
import traceback
import unittest

from cutoff import CutoffTest
from mfcc import MFCCTest
import signal_comparison
from sound_file import SoundFileTest
# the visualization is not very useful at the moment
# import visualization

# TODO:  implement tests for all packages


def run_tests():
    unittest.main()


if __name__ == "__main__":
    try:
        file_names = ["recordings//" + str + ".wav" for str in
                      ["ja1", "ja2", "ja3", "nein1", "nein2", "nein3"]]
        metrics = signal_comparison.compare_files(file_names, True)
        # nodes, edges = visualization.build_graph(metrics)
        # visualization.print_graph(nodes, edges)
    except Exception as e:
        traceback.print_exc(file=sys.stdout)
        print(repr(e))
