import numpy as np

from . import normalization
from .signal_type import Signal
from . import utility


# hard-elbow hiss reduction
def hiss_reduction(signal):
    signal = signal.copy()
    noisewindow = _determine_noise_window(signal)
    signal = Signal(_subtract_noise_from_signal(signal, noisewindow),
                    signal.sampleRate)
    return normalization.normalize(signal)


def _find_lowest_peak_volume_window(windows):
    max_value_per_window = {window.max(): i for i, window
                            in enumerate(windows)}
    min_max_value_key = min(max_value_per_window, key=max_value_per_window.get)
    window_with_lowest_peak = max_value_per_window[min_max_value_key]
    return windows[window_with_lowest_peak][:]


def _split_signal_into_windows(signal):
    samples_per_window = utility.compute_samples_per_window(signal.sampleRate)
    splittable_size = samples_per_window * (len(signal) // samples_per_window)
    window_range = range(0, splittable_size, samples_per_window)
    return np.array([signal[i:i+samples_per_window] for i in window_range])


def _determine_noise_window(signal):
    signal_windows = _split_signal_into_windows(signal)
    return _find_lowest_peak_volume_window(signal_windows)


def _subtract_noise_from_signal(signal, noise):
    noise_spectrum = np.fft.rfft(noise)
    noise_spectrum[0] = 0
    return _subtract_noise_frequencies(signal, noise_spectrum)


def _subtract_noise_frequencies(signal, noise_spectrum):
    windows = _split_signal_into_windows(signal)
    windows_no_noise = _remove_noise_from_signal_windows(windows,
                                                         noise_spectrum)
    return _combine_windows_to_signal(windows_no_noise)


def _remove_noise_from_signal_windows(windows, noise_spectrum):
    spectrums = np.array([np.fft.rfft(window) for window in windows])
    spectrums_no_noise = np.array([_remove_noise(window, noise_spectrum) for
                                   window in spectrums])
    return np.array([np.fft.irfft(window) for window in spectrums_no_noise])


def _remove_noise(spectrum, noise_spectrum):
    window_size = len(noise_spectrum)
    spectrum[0] = 0
    return np.asarray([_remove_noise_frequency(spectrum[i],
                                               noise_spectrum[i])
                       for i in range(window_size)])


def _remove_noise_frequency(signal, noise):
    if abs(signal) < abs(noise):
        return 0
    else:
        return signal


def _combine_windows_to_signal(windows):
    combination = []
    for window in windows:
        for element in window:
            combination.append(element)
    return np.array(combination)
