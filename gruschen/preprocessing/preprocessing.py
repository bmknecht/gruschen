from . import (
    cutoff,
    normalization,
    trimming,
)


def process(signal, advanced=False):
    signal = normalization.normalize(signal)
    if advanced:
        signal = _advanced_preprocessing(signal)
    # signal = trimming.smart_trim(signal, 0.1)
    return signal


def _advanced_preprocessing(signal):
    signal = trimming.trim_all_zeros(signal)
    # signal = hiss_reduction.hiss_reduction(signal)
    # signal = normalization.normalize_zero(signal)
    signal = cutoff.cutoff(signal)
    signal = normalization.normalize(signal)
    return signal
