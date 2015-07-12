
from signal_type import Signal


def trim_all_zeros(signal):
    return Signal([x for x in signal if abs(x) > 1e-10],
                  signal.sampleRate)


def smart_trim(signal, cutoffvalue):
    start = 0
    for i in range(len(signal)):
        if abs(signal[i]) > cutoffvalue:
            start = max(i-signal.sampleRate/2, 0)
            break

    end = len(signal) - 1
    for i in reversed(range(len(signal))):
        if abs(signal[i]) > cutoffvalue:
            end = min(i+signal.sampleRate/2, len(signal))
            break

    return signal[start:end]
