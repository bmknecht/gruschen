import math
import unittest

import numpy as np

from . signal_type import Signal
from . import utility


# unit test
class MFCCTest(unittest.TestCase):
    maxDiff = None
    commonSampleRates = [8000, 16000, 32000, 44100]
    testSize = 100

    def sine(self):
        return np.array([math.sin(x) for x in range(self.testSize)])

    def test_pre_emphasize(self):
        test_signal = _pre_emphasize(np.array([_ for _ in
                                               range(self.testSize)]))
        predictor_index = self.testSize - 1
        predictor = ((predictor_index - test_signal[predictor_index-1]) /
                     predictor_index)
        for i in range(2, len(test_signal)):
            self.assertLess(abs(predictor - (i+1 - test_signal[i]) / i), 0.1)

    def test_split_into_frame(self):
        frames = _split_into_frame(Signal([_ for _ in
                                           range(self.commonSampleRates[1])],
                                          self.commonSampleRates[0]))

        for n, frame in enumerate(frames):
            self.assertEqual(len(frame), len(frames[0]))
            for i, v in enumerate(frame):
                self.assertEqual(v, n * len(frame) + i)

    def is_power_of_two(self, n):
        return n == 2 ** int(math.log2(n))

    def test_frame_characteristics(self):
        for sample_rate in self.commonSampleRates:
            frame_size, frame_overlap = _frame_characteristics(sample_rate)
            self.assertEqual(frame_size, frame_overlap * 2)
            self.assertTrue(self.is_power_of_two(frame_size))
            self.assertTrue(self.is_power_of_two(frame_overlap))

    def test_window(self):
        impactful_number = 100
        not_just_one = 10
        windows = _window([np.array([impactful_number for _ in
                                     range(self.testSize)])
                           for _ in range(not_just_one)])
        for window in windows:
            for i in range(len(window) // 2 - 1):
                self.assertLess(window[i], window[i+1])
                self.assertAlmostEqual(window[i], window[len(window) - 1 - i])

    def test_fft(self):
        a_lot = 12
        signals = [Signal([0j for _ in range(utility.window_size())], 2)
                   for _ in range(a_lot)]
        some_frequencies = [1, 15, 23]
        for i in range(len(signals)):
            for f in some_frequencies:
                signals[i][f] = 2j
        for i in range(len(signals)):
            signals[i] = Signal(np.fft.ifft(signals[i]), 1)
        signals = _fft(signals)
        for signal in signals:
            for i in range(len(signal)):
                if i in some_frequencies:
                    self.assertAlmostEqual(signal[i], 4)
                else:
                    self.assertAlmostEqual(signal[i], 0)

    def test_mel_filtering(self):
        for sample_rate in self.commonSampleRates:
            not_just_one = 2
            windows = [Signal([1 for _ in range(utility.window_size())],
                              sample_rate)
                       for _ in range(not_just_one)]
            signal = _mel_filtering(windows)
            for i in range(len(signal[0])-1):
                self.assertLessEqual(signal[0][i]-1, signal[0][i+1])
                self.assertEqual(signal[0][i], signal[1][i])

    def test_non_linear_transform(self):
        not_just_one = 2
        signal = [self.sine() for _ in range(not_just_one)]
        transformed = _non_linear_transform(signal)
        for i in range(len(signal)):
            if signal[0][i] < logarithm_smallest_argument:
                self.assertEqual(transformed[0][i], logarithm_bottom_line)
            else:
                self.assertGreater(signal[0][i], transformed[0][i])
            self.assertEqual(transformed[0][i], transformed[1][i])


logarithm_smallest_argument = 1e-22
logarithm_bottom_line = -50
lowest_useful_frequency = 300
mel_bins_count = 25
useful_mel_bins_count = 13


# MFCC
def mfcc(signal):
    return (_mfcc_besides_pre_emphasizing(signal) +
            _mfcc_besides_pre_emphasizing(_pre_emphasize(signal)))


def _mfcc_besides_pre_emphasizing(signal):
    signal = _split_into_frame(signal)
    signal = _window(signal)
    signal = _fft(signal)
    signal = _mel_filtering(signal)
    signal = _non_linear_transform(signal)
    signal = _cepstral_coefficients(signal)
    return _zero_highest_frequency(signal)


def _pre_emphasize(signal):
    return signal - 0.97 * np.append(np.array([0]), signal[0:len(signal)-1])


def _split_into_frame(signal):
    frame_size, frame_overlap = _frame_characteristics(signal.sampleRate)
    return [_frame(signal, frame_size, i) for i in
            range(frame_overlap,
                  len(signal)-frame_overlap,
                  int(frame_size))]


def _frame_characteristics(sample_rate):
    frame_size = 128
    lowest_time_per_frame = 15
    while utility.sample_count_to_time(frame_size,
                                       sample_rate) < lowest_time_per_frame:
        frame_size *= 2
    return frame_size, frame_size // 2


def _frame(signal, frame_size, i):
    assert frame_size % 2 == 0
    return signal[i-frame_size//2:i+frame_size//2]


def _window(signal):
    n = len(signal[0])
    hamming_weights = 0.54 - 0.46 * np.cos(2*math.pi*np.arange(n) / (n-1))
    return [frame * hamming_weights for frame in signal]


def _fft(signal):

    def abs_square(c): return c.real*c.real + c.imag*c.imag

    def abs_square_array(x): return np.fromiter((abs_square(c) for c in x),
                                                np.float,
                                                len(x))
    return [Signal(abs_square_array(np.fft.fft(frame)), signal[0].sampleRate)
            for frame in signal]


def _mel_filtering(signal):
    center_of_bins = _mel_bin_center(signal[0].sampleRate, len(signal[0]) * 2)
    return [np.array([_mel_filter_window_sum(frame, i, center_of_bins[i-1:i+2])
                      for i in range(1, len(center_of_bins)-1)])
            for frame in signal]


def _mel_bin_center(samplerate, fft_size):
    lowest_uselful_mel_frequency = _mel_transform(lowest_useful_frequency)
    center_mel_frequencies = np.linspace(lowest_uselful_mel_frequency,
                                         _mel_transform(samplerate / 2 - 1),
                                         mel_bins_count)
    center_frequencies = _mel_transform_invert(center_mel_frequencies)
    center_spectrum_indices = ((fft_size / 2 + 1) /
                               samplerate * center_frequencies)
    return np.round(center_spectrum_indices).astype(int)


def _mel_transform(frequency):
    return 2595 * math.log10(1 + frequency / 700)


def _mel_transform_invert(mel_frequencies):
    return np.array([(10**(f / 2595)-1) * 700 for f in mel_frequencies])


def _mel_filter_window_sum(signal, bin_index, center_of_bins):
    a, b, c = center_of_bins

    def l1(x, y): return x / (b-a+1) * y

    def l2(x, y): return (1 - x / (c-b+1)) * y

    return (sum(map(l1, range(1, b-a+2), signal[a:b+1])) +
            sum(map(l2, range(0, c+1-b), signal[b:c+1])))


def _non_linear_transform(signal):
    return [np.array([math.log(s)
                      if s > logarithm_smallest_argument
                      else logarithm_bottom_line
                      for s in frame])
            for frame in signal]


def _cepstral_coefficients(signal):
    return [np.array([_cepstral_single_coefficient(frame, i)
                      for i in range(useful_mel_bins_count)])
            for frame in signal]


def _cepstral_single_coefficient(signal, index):
    cosine_sum = 0
    for f, freq in enumerate(signal):
        cosine_sum += freq * math.cos(math.pi * index /
                                      (mel_bins_count-2) * (f - 0.5))
    return cosine_sum


def _zero_highest_frequency(signal):
    signal = [np.fft.rfft(frame) for frame in signal]
    for i in range(len(signal)):
        signal[i][-1] = 0
    return [np.fft.irfft(frame) for frame in signal]
