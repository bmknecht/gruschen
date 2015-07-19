import statistics
import unittest

import numpy as np

from .signal_type import Signal


class CutoffTest(unittest.TestCase):
    maxDiff = None

    def test_cutoff(self):
        magic_number = 101
        signal = Signal([_ for _ in range(1000)], magic_number)
        signal = cutoff(signal)
        self.assertEqual(signal.sampleRate, magic_number)
        for i in range(len(signal) // 2):
            self.assertEqual(signal[i], 0)
        for i in range(len(signal) // 2 + 1, len(signal)):
            self.assertEqual(signal[i], i)

    def test_cutoff_frequency(self):
        signal = [0 for _ in range(128)]
        signal[0] = 1
        magic_number = 101
        cutoff_signal = cutoff_frequency(Signal(np.fft.irfft(signal),
                                                magic_number))
        self.assertEqual(cutoff_signal.sampleRate, magic_number)
        cutoff_spectrum = np.fft.rfft(cutoff_signal)
        self.assertEqual(len(cutoff_spectrum), len(signal))
        for i in range(len(cutoff_spectrum)):
            self.assertAlmostEqual(cutoff_spectrum[i], signal[i])


def cutoff(signal):
    median = statistics.median(abs(signal))
    # copy complex signal
    signal = Signal([x.real for x in signal], signal.sampleRate)
    for i, x in enumerate(signal):
        if abs(x) < median:
            signal[i] = 0
    return signal


def cutoff_frequency(signal):
    spectrum = np.fft.rfft(signal)
    median = statistics.median(abs(spectrum))
    for i, x in enumerate(spectrum):
        if abs(x) < median:
            spectrum[i] = 0
    return Signal(np.fft.irfft(spectrum), signal.sampleRate)
