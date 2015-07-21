import math
import unittest

import numpy as np
import scipy.fftpack as scipyfft
import scipy.signal

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
        signal = np.array([_ for _ in range(self.testSize)])
        pre_signal = _pre_emphasize(signal)
        for i in range(1, len(signal)):
            self.assertEqual(pre_signal[i], signal[i] - 0.97 * signal[i-1])

    def test_split_into_frame(self):
        frames = _split_into_frame(Signal([_ for _ in
                                           range(self.commonSampleRates[1])],
                                          self.commonSampleRates[0]))
        framesize = len(frames[0])
        self.assertTrue(framesize % 2 == 0)
        for i in range(1, len(frames)-1):
            self.assertEqual(len(frames[i-1]), len(frames[i]))
            self.assertEqual(len(frames[i]), len(frames[i+1]))
            self.assertTrue((frames[i-1][framesize//2:].asnparray() ==
                             frames[i][0:framesize//2].asnparray()).all())
            self.assertTrue((frames[i][framesize//2:].asnparray() ==
                             frames[i+1][0:framesize//2].asnparray()).all())

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
                weight = 0.54 - 0.46 * np.cos(2*math.pi*i / (self.testSize-1))
                self.assertAlmostEqual(window[i], weight * impactful_number)
                self.assertAlmostEqual(window[i], window[len(window) - 1 - i])
        for i in range(1, len(windows)):
            self.assertTrue((windows[i-1] == windows[i]).all())

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

    def test_mel_transformations(self):
        test = [50, 100, 200, 300, 400, 500]
        transformed = [_mel_transform(t) for t in test]
        test2 = _mel_transform_invert(transformed)
        self.assertEqual(len(test), len(transformed))
        validation = [77.75568921108743,
                      150.49127889683163,
                      283.2339944477894,
                      401.97639977236383,
                      509.39197126252225,
                      607.4547050090661]
        for i in range(len(transformed)):
            self.assertTrue(abs(transformed[i] - validation[i]) < 1e-2)
        self.assertEqual(len(test), len(test2))
        for i in range(len(test)):
            self.assertAlmostEqual(test2[i], test[i])

    def test_mel_bin_center(self):
        global mel_bins_count
        old_mel_bins_count = mel_bins_count
        mel_bins_count = 12
        centers = _mel_bin_center(16e3, 512)
        validation = [9, 16, 25, 35, 47, 63, 81, 104, 132, 165, 206, 256]
        for i in range(len(centers)):
            self.assertLess(abs(centers[i] - validation[i]), 2)
        mel_bins_count = old_mel_bins_count

    def test_mel_filtering(self):
        signal = [Signal(abs(self.sine()), 8000) for _ in range(3)]
        mel_sums = _mel_filtering(signal)
        self.assertEqual(len(mel_sums), len(signal))
        for i in range(3):
            for j in range(len(mel_sums[i])):
                self.assertLess(mel_sums[i][j], sum(signal[i]))

    def test_mel_filter_window_sum(self):
        signal = np.array(list(range(1, 11)))
        bin_centers = [0, 4, 10]
        s = _mel_filter_window_sum(signal, bin_centers)

        self.assertAlmostEqual(s, 88/3)

    def test_cepstral_coefficients(self):
        pass

    def test_cepstral_single_coefficient(self):
        pass

    def test_zero_highest_frequency(self):
        spectrum = [9, 9, 9, 9, 9, 9, 9]
        zeroed_signal = _zero_highest_frequency([np.fft.irfft(spectrum)])
        zeroed_spectrum = np.fft.rfft(zeroed_signal[0])
        self.assertAlmostEqual(zeroed_spectrum[-1], 0)
        for i in range(len(spectrum)-1):
            self.assertAlmostEqual(spectrum[i], 9)

    def test_compute_deltas(self):
        signal = [list(range(1, 11)), list(range(11, 21))]
        delta = _compute_deltas(signal)
        self.assertEqual(len(delta), 2)
        self.assertEqual(len(delta[0]), 10)
        self.assertEqual(len(delta[1]), 10)
        for i in range(len(delta)):
            self.assertEqual(delta[0][i], 1)
            self.assertEqual(delta[0][i], 1)

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
lowest_useful_frequency = 200
mel_bins_count = 25
useful_mel_bins_count = 13


# MFCC
def mfcc(signal):
    coefficients = _mfcc_besides_pre_emphasizing(_pre_emphasize(signal))
    delta = _compute_deltas(np.array(coefficients))
    return _combine_signals([coefficients,
                             delta,
                             _compute_deltas(delta),
                             _mfcc_besides_pre_emphasizing(signal)])


def _mfcc_besides_pre_emphasizing(signal):
    signal = _split_into_frame(signal)
    signal = _window(signal)
    signal = _fft(signal)
    signal = _mel_filtering(signal)
    signal = _non_linear_transform(signal)
    signal = _cepstral_coefficients(signal)
    return _zero_highest_frequency(signal)


def _pre_emphasize(signal):
    return signal - 0.97 * np.append(np.array([0]), signal[0:-1])


def _split_into_frame(signal):
    frame_size, frame_overlap = _frame_characteristics(signal.sampleRate)
    return [signal[i-frame_size//2:i+frame_size//2] for i in
            range(frame_size//2,
                  len(signal)-frame_size//2,
                  int(frame_overlap))]


def _frame_characteristics(sample_rate):
    frame_size = 128
    lowest_time_per_frame = 20
    while utility.sample_count_to_time(frame_size,
                                       sample_rate) < lowest_time_per_frame:
        frame_size *= 2
    return frame_size, frame_size // 2


def _window(signal):
    return [frame * scipy.signal.hamming(len(signal[0])) for frame in signal]


def _fft(signal):

    def abs_square(c): return c.real*c.real + c.imag*c.imag

    def abs_square_array(x): return np.fromiter((abs_square(c) for c in x),
                                                np.float,
                                                len(x))
    return [Signal(abs_square_array(np.fft.fft(frame)), signal[0].sampleRate)
            for frame in signal]


def _mel_filtering(signal):
    center_of_bins = _mel_bin_center(signal[0].sampleRate, len(signal[0]) * 2)
    return [np.array([_mel_filter_window_sum(frame, center_of_bins[i-1:i+2])
                      for i in range(1, len(center_of_bins)-1)])
            for frame in signal]


def _mel_bin_center(samplerate, fft_size):
    lowest_uselful_mel_frequency = _mel_transform(lowest_useful_frequency)
    center_mel_frequencies = np.linspace(lowest_uselful_mel_frequency,
                                         _mel_transform(samplerate / 2 - 1),
                                         mel_bins_count)
    center_frequencies = _mel_transform_invert(center_mel_frequencies)
    center_spectrum_indices = ((fft_size-1) /
                               samplerate * center_frequencies)
    return np.round(center_spectrum_indices).astype(int)


def _mel_transform(frequency):
    return 1127.01048 * math.log(1 + float(frequency) / 700.)


def _mel_transform_invert(mel_frequencies):
    return np.array([(math.exp(f / 1127.01048)-1)*700 for f in mel_frequencies])


def _mel_filter_window_sum(signal, center_of_bins):
    a, b, c = center_of_bins

    def l1(x, y): return x / (b-a+1) * y

    def l2(x, y): return (1 - x / (c-b)) * y

    return (sum(map(l1, range(1, b-a+2), signal[a:b+1])) +
            sum(map(l2, range(1, c-b), signal[b+1:c+1])))


def _non_linear_transform(signal):
    return [np.array([math.log(s)
                      if s > logarithm_smallest_argument
                      else logarithm_bottom_line
                      for s in frame])
            for frame in signal]


def _cepstral_coefficients(signal):
    return [scipyfft.dct(frame)[:useful_mel_bins_count] for frame in signal]


def _zero_highest_frequency(signal):
    signal = [np.fft.fft(frame) for frame in signal]
    for i in range(len(signal)):
        signal[i][-1] = 0
        signal[i][len(signal[i])//2] = 0
    return [np.real(np.fft.ifft(frame)) for frame in signal]


def _compute_deltas(vec):
    npcoeff = np.array(vec)
    npdelta = npcoeff - np.array([np.append(0, k[0:-1]) for k in npcoeff])
    return npdelta.tolist()


def _combine_signals(vecs):
    t = [frame.tolist() for frame in vecs[0]]
    for i in range(1, len(vecs)):
        assert len(t) == len(vecs[i])
        for j in range(len(t)):
            t[j].extend(vecs[i][j])
    return [np.array(frame) for frame in t]
