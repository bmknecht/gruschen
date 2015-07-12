from signal_type import Signal


def normalize(signal):
    return Signal(signal[:] / max(abs(signal.max()), abs(signal.min())),
                  signal.sampleRate)


def normalize_zero(signal):
    return normalize(signal - signal.mean())
