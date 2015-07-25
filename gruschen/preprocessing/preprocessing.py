from . import (
    cutoff,
    normalization,
    trimming,
)


def process(signal, samplerate, advanced=True):
    signal = normalization.normalize(signal)
    if advanced:
        signal = _advanced_preprocessing(signal)
    signal = trimming.smart_trim(signal, samplerate, 0.1)
    return signal


def _advanced_preprocessing(signal):
    signal = cutoff.cutoff(signal)
    signal = normalization.normalize(signal)
    return signal
