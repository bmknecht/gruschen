
def smart_trim(signal, samplerate, cutoffvalue):
    tolerance = samplerate/200   # 5 ms
    start = 0
    for i in range(len(signal)):
        if abs(signal[i]) > cutoffvalue:
            start = max(i-tolerance, 0)
            break

    end = len(signal) - 1
    for i in reversed(range(len(signal))):
        if abs(signal[i]) > cutoffvalue:
            end = min(i+tolerance, len(signal))
            break

    return signal[start:end]
