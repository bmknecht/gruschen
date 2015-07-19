from . import (
    cutoff,
    hiss_reduction,
    normalization,
    trimming,
)


def process(signal, advanced=False):
    signal = normalization.normalize(signal)
    if advanced:
        signal = _advanced_pre_processing(signal)
    signal = trimming.smart_trim(signal, 0.1)
    return signal


def _advanced_pre_processing(signal):
    signal = trimming.trim_all_zeros(signal)
    signal = hiss_reduction.hiss_reduction(signal)
    signal = normalization.normalize_zero(signal)
    signal = cutoff.cutoff(signal)
    signal = normalization.normalize(signal)
